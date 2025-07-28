import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import csv
from tqdm.auto import tqdm
from PIL import Image
import numba
from numba import jit, prange
import os
import subprocess
import random
from multiprocessing.pool import ThreadPool
import time

# PyMOO imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation

#----------------------- Configuration Variables --------------------------#
POPULATION = 20
GENERATIONS = 50
SEED = 42
MUTATION_RATE = 0.15
NUM_PES = 25  # Number of Processing Elements
EVAL_SAMPLES = 1000  
#----------------------- Configuration Variables --------------------------#

# Global variables
pool = ThreadPool(4) 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Global model and data (loaded once)
global_model = None
global_dataloader = None
global_weights = None

#----------------------- PE Configuration Handler --------------------------#
class PEConfig:
    def __init__(self, pe_config):
        """
        Initialize PE configuration
        pe_config: List of tuples, each tuple contains 4 values representing
                  which Booth variant to use for each partial product in a multiplier
        """
        self.pe_config = pe_config
        self.num_pes = len(pe_config)
    
    def get_pe_variants(self, pe_index):
        """Get the 4 variants for a specific PE"""
        if pe_index < len(self.pe_config):
            return self.pe_config[pe_index]
        else:
            # Default to exact multiplier if PE index exceeds config
            return (0, 0, 0, 0)

#----------------------- Booth Multiplier Implementation --------------------------#
def sign_extend(val, bits=8):
    """Properly handle sign extension for 8-bit values"""
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val
@jit(nopython=True, cache=True)
def booth_mult_with_pe_config(A, B, pp0_variant, pp1_variant, pp2_variant, pp3_variant):
    """
    Booth multiplier with configurable variants for each partial product
    Variants: 0=Exact, 1=Approx1, 2=Approx2, 3=Approx3
    """

    A = max(-128, min(127, A))
    B = max(-128, min(127, B))


    # Sign extend
    if A >= 128:
        A = A - 256
    if B >= 128:
        B = B - 256
    
    block = (B << 1) & 0x1FF
    # block = B << 1
    Product = 0
    # print(block)
    variants = np.array([pp0_variant, pp1_variant, pp2_variant, pp3_variant])
    # print(variants)
    for i in range(4):
        code = block & 0b111
        variant = variants[i]
        
        # Select partial product based on variant
        if variant == 0:  # Exact
            if code == 0 or code == 7:
                PP = 0
            elif code == 1 or code == 2:
                PP = A
            elif code == 3:
                PP = A << 1
            elif code == 4:
                PP = -(A << 1)
            else:  
                PP = -A
        
        elif variant == 1:  # Approximation 1
            if code == 0 or code == 7:
                PP = 0
            elif code == 1 or code == 2 or code == 5 or code == 6:
                PP = A
            elif code == 3 or code == 4:
                PP = A << 1
            else:
                PP = 0
        
        elif variant == 2:  # Approximation 2
            if code == 0 or code == 7:
                PP = 0
            elif code == 1 or code == 2 or code == 3:
                PP = A
            elif code == 5 or code == 6 or code == 4:
                PP = -A
            else:
                PP = 0
        
        elif variant == 3:  # Approximation 3
            if code == 0 or code == 7:
                PP = 0
            elif code == 1 or code == 2 or code == 5 or code == 6:
                PP = -A
            elif code == 3 or code == 4:
                PP = -(A << 1)
            else:
                PP = 0
        else:
            PP = 0

        Product += PP << (2 * i)
        block >>= 2

    return Product

@jit(nopython=True, parallel=True, cache=True)
def vectorized_booth_conv_with_pe(input_matrix, kernel, pe_variants, stride=1, padding=0):
    """Convolution with PE configuration"""
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width), dtype=np.int32)
    
    pp0_var, pp1_var, pp2_var, pp3_var = pe_variants

    for i in prange(output_height):
        for j in prange(output_width):
            # acc = 0
            acc = int(0)
            for m in range(kernel_height):
                for n in range(kernel_width):
                    x = i * stride + m - padding
                    y = j * stride + n - padding
                    
                    if 0 <= x < input_height and 0 <= y < input_width:
                        a = input_matrix[x, y]
                        b = kernel[m, n]
                        acc += booth_mult_with_pe_config(a, b, pp0_var, pp1_var, pp2_var, pp3_var)
            
            output_matrix[i, j] = acc
    
    return output_matrix

@jit(nopython=True, cache=True)
def max_pool_2x2(input_array):
    """Fast 2x2 max pooling"""
    h, w = input_array.shape
    out_h, out_w = h // 2, w // 2
    output = np.zeros((out_h, out_w), dtype=input_array.dtype)
    
    for i in range(out_h):
        for j in range(out_w):
            window = input_array[i*2:(i+1)*2, j*2:(j+1)*2]
            output[i, j] = np.max(window)
    
    return output

#----------------------- Model and Data Setup --------------------------#
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@jit(nopython=True, cache=True)
def quantize_weights_fast(weights, num_bits=8):
    """Fast weight quantization"""
    flat_weights = weights.flatten()
    max_val = 0.0
    for val in flat_weights:
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
    
    if max_val == 0:
        return np.zeros_like(weights, dtype=np.int8), 1.0
    
    scale = max_val / 127.0
    quantized = np.zeros_like(weights, dtype=np.int8)
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            for k in range(weights.shape[2]):
                for l in range(weights.shape[3]):
                    val = weights[i, j, k, l]
                    quantized[i, j, k, l] = max(-128, min(127, int(val / scale)))
    
    return quantized, scale

def preprocess_model_weights(model):
    """Preprocess and quantize all model weights once"""
    conv1_w = model.features[0].weight.data.numpy()
    conv1_b = model.features[0].bias.data.numpy()
    conv2_w = model.features[3].weight.data.numpy()
    conv2_b = model.features[3].bias.data.numpy()
    
    # Quantize weights
    conv1_w_q, conv1_scale = quantize_weights_fast(conv1_w)
    conv2_w_q, conv2_scale = quantize_weights_fast(conv2_w)
    
    return {
        'conv1_w_q': conv1_w_q,
        'conv1_b': conv1_b,
        'conv1_scale': conv1_scale,
        'conv2_w_q': conv2_w_q,
        'conv2_b': conv2_b,
        'conv2_scale': conv2_scale
    }

def setup_global_data():
    """Setup global model and data"""
    global global_model, global_dataloader, global_weights
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    global_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Setup model
    global_model = LeNet5()
    
    global_model.eval()
    global_weights = preprocess_model_weights(global_model)
    
    print(" Global data and model setup complete")

#----------------------- Generate Verilog Code --------------------------#    

def generate_verilog_from_combs(comb_list):

    print("entered verilog generateion")

    assert len(comb_list) == 25, "Expected exactly 25 comb values"

    param_lines = []

    for i, comb in enumerate(comb_list):

        value = (int(comb[3]) << 6) | (int(comb[2]) << 4) | (int(comb[1]) << 2) | int(comb[0])

        param_lines.append(f"    parameter [7:0] comb{i} = 8'b{value:08b},")
    param_block = "\n".join(param_lines).rstrip(",")

    verilog = f"""
// Auto-generated systolic array with heterogeneous PEs
module systolic_array #(
{param_block}
,parameter N = 8
)(
    input wire clk,                // Clock signal
    input wire reset,              // Reset signal
    input wire control,            // Control signal (1: load weights, 0: data flow)
    
    input wire  [N-1:0] data_in_row_0,  // Data input for row 0
    input wire  [N-1:0] data_in_row_1,  // Data input for row 1
    input wire  [N-1:0] data_in_row_2,  // Data input for row 2
    input wire  [N-1:0] data_in_row_3,  // Data input for row 3
    input wire  [N-1:0] data_in_row_4,  // Data input for row 4
    
    input wire  [N-1:0] weight_in_col_0, // Weight input for column 0
    input wire  [N-1:0] weight_in_col_1, // Weight input for column 1
    input wire  [N-1:0] weight_in_col_2, // Weight input for column 2
    input wire  [N-1:0] weight_in_col_3, // Weight input for column 3
    input wire  [N-1:0] weight_in_col_4, // Weight input for column 4 
    
    output wire  [2*N-1:0] acc_out_0,  // Accumulation output for row 0
    output wire  [2*N-1:0] acc_out_1,  // Accumulation output for row 1
    output wire  [2*N-1:0] acc_out_2,   // Accumulation output for row 2
    output wire  [2*N-1:0] acc_out_3,   // Accumulation output for row 3
    output wire  [2*N-1:0] acc_out_4   // Accumulation output for row 4
);
    


    // Internal wires to connect each PE
    wire  [(N/2)-1:0] neg_in_00,  neg_in_10,  neg_in_20, neg_in_30, neg_in_40;
    wire  [(N/2)-1:0] neg_out_00, neg_out_01, neg_out_02, neg_out_03, neg_out_04;    
    wire  [(N/2)-1:0] neg_out_10, neg_out_11, neg_out_12, neg_out_13, neg_out_14;    
    wire  [(N/2)-1:0] neg_out_20, neg_out_21, neg_out_22, neg_out_23, neg_out_24;
    wire  [(N/2)-1:0] neg_out_30, neg_out_31, neg_out_32, neg_out_33, neg_out_34;
    wire  [(N/2)-1:0] neg_out_40, neg_out_41, neg_out_42, neg_out_43, neg_out_44;
    
    wire  [(N/2)-1:0] one_in_00,  one_in_10,  one_in_20, one_in_30, one_in_40;
    wire  [(N/2)-1:0] one_out_00, one_out_01, one_out_02, one_out_03, one_out_04;    
    wire  [(N/2)-1:0] one_out_10, one_out_11, one_out_12, one_out_13, one_out_14;    
    wire  [(N/2)-1:0] one_out_20, one_out_21, one_out_22, one_out_23, one_out_24;
    wire  [(N/2)-1:0] one_out_30, one_out_31, one_out_32, one_out_33, one_out_34;
    wire  [(N/2)-1:0] one_out_40, one_out_41, one_out_42, one_out_43, one_out_44;
    
    wire  [(N/2)-1:0] two_in_00,  two_in_10,  two_in_20, two_in_30, two_in_40;
    wire  [(N/2)-1:0] two_out_00, two_out_01, two_out_02, two_out_03, two_out_04;    
    wire  [(N/2)-1:0] two_out_10, two_out_11, two_out_12, two_out_13, two_out_14;    
    wire  [(N/2)-1:0] two_out_20, two_out_21, two_out_22, two_out_23, two_out_24;
    wire  [(N/2)-1:0] two_out_30, two_out_31, two_out_32, two_out_33, two_out_34;
    wire  [(N/2)-1:0] two_out_40, two_out_41, two_out_42, two_out_43, two_out_44;
    
    wire  [N-1:0] B1_in_00,  B1_in_01,  B1_in_02, B1_in_03, B1_in_04;
    wire  [N-1:0] B1_out_00, B1_out_01, B1_out_02, B1_out_03, B1_out_04;    
    wire  [N-1:0] B1_out_10, B1_out_11, B1_out_12, B1_out_13, B1_out_14;    
    wire  [N-1:0] B1_out_20, B1_out_21, B1_out_22, B1_out_23, B1_out_24;
    wire  [N-1:0] B1_out_30, B1_out_31, B1_out_32, B1_out_33, B1_out_34;
    wire  [N-1:0] B1_out_40, B1_out_41, B1_out_42, B1_out_43, B1_out_44;
    
    
    wire  [N-1:0] B2_in_00,  B2_in_01,  B2_in_02, B2_in_03, B2_in_04;
    wire  [N-1:0] B2_out_00, B2_out_01, B2_out_02, B2_out_03, B2_out_04;    
    wire  [N-1:0] B2_out_10, B2_out_11, B2_out_12, B2_out_13, B2_out_14;    
    wire  [N-1:0] B2_out_20, B2_out_21, B2_out_22, B2_out_23, B2_out_24;
    wire  [N-1:0] B2_out_30, B2_out_31, B2_out_32, B2_out_33, B2_out_34;
    wire  [N-1:0] B2_out_40, B2_out_41, B2_out_42, B2_out_43, B2_out_44;
    
    wire  [2*N-1:0] acc_out_internal_00, acc_out_internal_01, acc_out_internal_02, acc_out_internal_03, acc_out_internal_04; 
    wire  [2*N-1:0] acc_out_internal_10, acc_out_internal_11, acc_out_internal_12, acc_out_internal_13, acc_out_internal_14;
    wire  [2*N-1:0] acc_out_internal_20, acc_out_internal_21, acc_out_internal_22, acc_out_internal_23, acc_out_internal_24;
    wire  [2*N-1:0] acc_out_internal_30, acc_out_internal_31, acc_out_internal_32, acc_out_internal_33, acc_out_internal_34;
    wire  [2*N-1:0] acc_out_internal_40, acc_out_internal_41, acc_out_internal_42, acc_out_internal_43, acc_out_internal_44;
    
    //booth control signals generation for row-0
        
    booth_control_top #(.N(N)) control_0 (    
    .A(data_in_row_0),
    .neg(neg_in_00),
    .one(one_in_00),
    .two(two_in_00)    
    );
    
    //booth control signals generation for row-1
    
    booth_control_top #(.N(N)) control_1(    
    .A(data_in_row_1),
    .neg(neg_in_10),
    .one(one_in_10),
    .two(two_in_10)    
    );
    
    //booth control signals generation for row-2
    
    booth_control_top #(.N(N)) control_2( 
    .A(data_in_row_2),
    .neg(neg_in_20),
    .one(one_in_20),
    .two(two_in_20)    
    );
    
    //booth control signals generation for row-3
    
    booth_control_top #(.N(N)) control_3( 
    .A(data_in_row_3),
    .neg(neg_in_30),
    .one(one_in_30),
    .two(two_in_30)    
    );
    
     //booth control signals generation for row-4
    
    booth_control_top #(.N(N)) control_4( 
    .A(data_in_row_4),
    .neg(neg_in_40),
    .one(one_in_40),
    .two(two_in_40)    
    );
    
    assign B1_in_00 = weight_in_col_0;
    assign B2_in_00 = weight_in_col_0 << 1; 
    
    assign B1_in_01 = weight_in_col_1;
    assign B2_in_01 = weight_in_col_1 << 1; 
    
    assign B1_in_02 = weight_in_col_2;
    assign B2_in_02 = weight_in_col_2 << 1; 
    
    assign B1_in_03 = weight_in_col_3;
    assign B2_in_03 = weight_in_col_3 << 1;
    
    assign B1_in_04 = weight_in_col_4;
    assign B2_in_04 = weight_in_col_4 << 1; 
    
    
    // Row 0 PE instantiations
           
           
       PE #(.N(N), .comb(comb0)) pe_00(
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in('b0),      
        .B1_in(B1_in_00),  
        .B2_in(B2_in_00),  
        .neg_in(neg_in_00), 
        .one_in(one_in_00), 
        .two_in(two_in_00), 
        .B1_out(B1_out_00), 
        .B2_out(B2_out_00),
        .neg_out(neg_out_00),
        .one_out(one_out_00),
        .two_out(two_out_00),
        .acc_out(acc_out_internal_00)
    );
    
           
       PE #(.N(N), .comb(comb1)) pe_01 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in('b0),      
        .B1_in(B1_in_01),  
        .B2_in(B2_in_01),  
        .neg_in(neg_out_00), 
        .one_in(one_out_00), 
        .two_in(two_out_00), 
        .B1_out(B1_out_01), 
        .B2_out(B2_out_01),
        .neg_out(neg_out_01),
        .one_out(one_out_01),
        .two_out(two_out_01),
        .acc_out(acc_out_internal_01)
    );
    
           
       PE #(.N(N), .comb(comb2)) pe_02 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in('b0),      
        .B1_in(B1_in_02),  
        .B2_in(B2_in_02),  
        .neg_in(neg_out_01), 
        .one_in(one_out_01), 
        .two_in(two_out_01), 
        .B1_out(B1_out_02), 
        .B2_out(B2_out_02),
        .neg_out(neg_out_02),
        .one_out(one_out_02),
        .two_out(two_out_02),
        .acc_out(acc_out_internal_02)
    );
    
           
       PE #(.N(N), .comb(comb3)) pe_03(
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in('b0),      
        .B1_in(B1_in_03),  
        .B2_in(B2_in_03),  
        .neg_in(neg_out_02), 
        .one_in(one_out_02), 
        .two_in(two_out_02), 
        .B1_out(B1_out_03), 
        .B2_out(B2_out_03),
        .neg_out(neg_out_03),
        .one_out(one_out_03),
        .two_out(two_out_03),
        .acc_out(acc_out_internal_03)
    );
    
           
       PE #(.N(N), .comb(comb4)) pe_04(
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in('b0),      
        .B1_in(B1_in_04),  
        .B2_in(B2_in_04),  
        .neg_in(neg_out_03),
        .one_in(one_out_03),
        .two_in(two_out_03), 
        .B1_out(B1_out_04), 
        .B2_out(B2_out_04),
        .neg_out(neg_out_04),
        .one_out(one_out_04),
        .two_out(two_out_04),
        .acc_out(acc_out_internal_04)
    );
    
    // Row 1 PE instantiations
    
      
     PE #(.N(N), .comb(comb5)) pe_10 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_00),      
        .B1_in(B1_out_00),  
        .B2_in(B2_out_00),  
        .neg_in(neg_in_10), 
        .one_in(one_in_10), 
        .two_in(two_in_10), 
        .B1_out(B1_out_10), 
        .B2_out(B2_out_10),
        .neg_out(neg_out_10),
        .one_out(one_out_10),
        .two_out(two_out_10),
        .acc_out(acc_out_internal_10)
    );
    
           
       PE #(.N(N), .comb(comb6)) pe_11 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_01),      
        .B1_in(B1_out_01),  
        .B2_in(B2_out_01),  
        .neg_in(neg_out_10), 
        .one_in(one_out_10), 
        .two_in(two_out_10), 
        .B1_out(B1_out_11), 
        .B2_out(B2_out_11),
        .neg_out(neg_out_11),
        .one_out(one_out_11),
        .two_out(two_out_11),
        .acc_out(acc_out_internal_11)
    );
    
           
       PE #(.N(N), .comb(comb7)) pe_12 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_02),      
        .B1_in(B1_out_02),  
        .B2_in(B2_out_02),  
        .neg_in(neg_out_11), 
        .one_in(one_out_11), 
        .two_in(two_out_11), 
        .B1_out(B1_out_12), 
        .B2_out(B2_out_12),
        .neg_out(neg_out_12),
        .one_out(one_out_12),
        .two_out(two_out_12),
        .acc_out(acc_out_internal_12)
    );
    
        
       PE #(.N(N), .comb(comb8)) pe_13 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_03),      
        .B1_in(B1_out_03),  
        .B2_in(B2_out_03),  
        .neg_in(neg_out_12), 
        .one_in(one_out_12), 
        .two_in(two_out_12), 
        .B1_out(B1_out_13), 
        .B2_out(B2_out_13),
        .neg_out(neg_out_13),
        .one_out(one_out_13),
        .two_out(two_out_13),
        .acc_out(acc_out_internal_13)
    );
    
        
       PE #(.N(N), .comb(comb9)) pe_14 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_04),      
        .B1_in(B1_out_04),  
        .B2_in(B2_out_04),  
        .neg_in(neg_out_13), 
        .one_in(one_out_13), 
        .two_in(two_out_13), 
        .B1_out(B1_out_14), 
        .B2_out(B2_out_14),
        .neg_out(neg_out_14),
        .one_out(one_out_14),
        .two_out(two_out_14),
        .acc_out(acc_out_internal_14)
    );
    
    // Row 2 PE instantiations

      
     PE #(.N(N), .comb(comb10)) pe_20 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_10),      
        .B1_in(B1_out_10),  
        .B2_in(B2_out_10),  
        .neg_in(neg_in_20), 
        .one_in(one_in_20), 
        .two_in(two_in_20), 
        .B1_out(B1_out_20), 
        .B2_out(B2_out_20),
        .neg_out(neg_out_20),
        .one_out(one_out_20),
        .two_out(two_out_20),
        .acc_out(acc_out_internal_20)
    );    
        
           
       PE #(.N(N), .comb(comb11)) pe_21 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_11),      
        .B1_in(B1_out_11),  
        .B2_in(B2_out_11),  
        .neg_in(neg_out_20), 
        .one_in(one_out_20), 
        .two_in(two_out_20), 
        .B1_out(B1_out_21), 
        .B2_out(B2_out_21),
        .neg_out(neg_out_21),
        .one_out(one_out_21),
        .two_out(two_out_21),
        .acc_out(acc_out_internal_21)
    );
    
           
       PE #(.N(N), .comb(comb12)) pe_22 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_12),      
        .B1_in(B1_out_12),  
        .B2_in(B2_out_12),  
        .neg_in(neg_out_21), 
        .one_in(one_out_21), 
        .two_in(two_out_21), 
        .B1_out(B1_out_22), 
        .B2_out(B1_out_22),
        .neg_out(neg_out_22),
        .one_out(one_out_22),
        .two_out(two_out_22),
        .acc_out(acc_out_internal_22)
    );
    
        
       PE #(.N(N), .comb(comb13)) pe_23 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_13),      
        .B1_in(B1_out_13),  
        .B2_in(B2_out_13),  
        .neg_in(neg_out_22), 
        .one_in(one_out_22), 
        .two_in(two_out_22), 
        .B1_out(B1_out_23), 
        .B2_out(B2_out_23),
        .neg_out(neg_out_23),
        .one_out(one_out_23),
        .two_out(two_out_23),
        .acc_out(acc_out_internal_23)
    );
    
        
       PE #(.N(N), .comb(comb14)) pe_24 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_14),      
        .B1_in(B1_out_14),  
        .B2_in(B2_out_14),  
        .neg_in(neg_out_23), 
        .one_in(one_out_23), 
        .two_in(two_out_23), 
        .B1_out(B1_out_24), 
        .B2_out(B2_out_24),
        .neg_out(neg_out_24),
        .one_out(one_out_24),
        .two_out(two_out_24),
        .acc_out(acc_out_internal_24)
    );
    
    
    // Row 3 PE instantiations

      
     PE #(.N(N), .comb(comb15)) pe_30 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_20),      
        .B1_in(B1_out_20),  
        .B2_in(B2_out_20),  
        .neg_in(neg_in_30), 
        .one_in(one_in_30), 
        .two_in(two_in_30), 
        .B1_out(B1_out_30), 
        .B2_out(B2_out_30),
        .neg_out(neg_out_30),
        .one_out(one_out_30),
        .two_out(two_out_30),
        .acc_out(acc_out_internal_30)
    );    
        
           
       PE #(.N(N), .comb(comb16)) pe_31 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_21),      
        .B1_in(B1_out_21),  
        .B2_in(B2_out_21),  
        .neg_in(neg_out_30), 
        .one_in(one_out_30), 
        .two_in(two_out_30), 
        .B1_out(B1_out_31), 
        .B2_out(B2_out_31),
        .neg_out(neg_out_31),
        .one_out(one_out_31),
        .two_out(two_out_31),
        .acc_out(acc_out_internal_31)
    );
    
           
       PE #(.N(N), .comb(comb17)) pe_32 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_22),      
        .B1_in(B1_out_22),  
        .B2_in(B2_out_22),  
        .neg_in(neg_out_31), 
        .one_in(one_out_31), 
        .two_in(two_out_31), 
        .B1_out(B1_out_32), 
        .B2_out(B2_out_32),
        .neg_out(neg_out_32),
        .one_out(one_out_32),
        .two_out(two_out_32),
        .acc_out(acc_out_internal_32)
    );
    
        
       PE #(.N(N), .comb(comb18)) pe_33 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_23),      
        .B1_in(B1_out_23),  
        .B2_in(B2_out_23),  
        .neg_in(neg_out_32), 
        .one_in(one_out_32), 
        .two_in(two_out_32), 
        .B1_out(B1_out_33), 
        .B2_out(B2_out_33),
        .neg_out(neg_out_33),
        .one_out(one_out_33),
        .two_out(two_out_33),
        .acc_out(acc_out_internal_33)
    );
    
        
       PE #(.N(N), .comb(comb19)) pe_34 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_24),      
        .B1_in(B1_out_24),  
        .B2_in(B2_out_24),  
        .neg_in(neg_out_33), 
        .one_in(one_out_33), 
        .two_in(two_out_33), 
        .B1_out(B1_out_34), 
        .B2_out(B2_out_34),
        .neg_out(neg_out_34),
        .one_out(one_out_34),
        .two_out(two_out_34),
        .acc_out(acc_out_internal_34)
    );
    
    // Row 4 PE instantiations

      
     PE #(.N(N), .comb(comb20)) pe_40 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_30),      
        .B1_in(B1_out_30),  
        .B2_in(B2_out_30),  
        .neg_in(neg_in_40), 
        .one_in(one_in_40), 
        .two_in(two_in_40), 
        .B1_out(B1_out_40), 
        .B2_out(B2_out_40),
        .neg_out(neg_out_40),
        .one_out(one_out_40),
        .two_out(two_out_40),
        .acc_out(acc_out_internal_40)
    );    
                   
       PE #(.N(N), .comb(comb21)) pe_41 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_31),      
        .B1_in(B1_out_31),  
        .B2_in(B2_out_31),  
        .neg_in(neg_out_40), 
        .one_in(one_out_40), 
        .two_in(two_out_40), 
        .B1_out(B1_out_41), 
        .B2_out(B2_out_41),
        .neg_out(neg_out_41),
        .one_out(one_out_41),
        .two_out(two_out_41),
        .acc_out(acc_out_internal_41)
    );
    
           
       PE #(.N(N), .comb(comb22)) pe_42 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_32),      
        .B1_in(B1_out_32),  
        .B2_in(B2_out_32),  
        .neg_in(neg_out_41), 
        .one_in(one_out_41), 
        .two_in(two_out_41), 
        .B1_out(B1_out_42), 
        .B2_out(B2_out_42),
        .neg_out(neg_out_42),
        .one_out(one_out_42),
        .two_out(two_out_42),
        .acc_out(acc_out_internal_42)
    );
    
        
       PE #(.N(N), .comb(comb23)) pe_43 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_33),      
        .B1_in(B1_out_33),  
        .B2_in(B2_out_33),  
        .neg_in(neg_out_42), 
        .one_in(one_out_42), 
        .two_in(two_out_42), 
        .B1_out(B1_out_43), 
        .B2_out(B2_out_43),
        .neg_out(neg_out_43),
        .one_out(one_out_43),
        .two_out(two_out_43),
        .acc_out(acc_out_internal_43)
    );
    
        
       PE #(.N(N), .comb(comb24)) pe_44 (
        .clk(clk),          
        .reset(reset),        
        .control(control),
        .acc_in(acc_out_internal_34),      
        .B1_in(B1_out_34),  
        .B2_in(B2_out_34),  
        .neg_in(neg_out_43), 
        .one_in(one_out_43), 
        .two_in(two_out_43), 
        .B1_out(B1_out_44), 
        .B2_out(B2_out_44),
        .neg_out(neg_out_44),
        .one_out(one_out_44),
        .two_out(two_out_44),
        .acc_out(acc_out_internal_44)
    );    

    // Final accumulation outputs (results from the last column)
    assign acc_out_0 = acc_out_internal_40;
    assign acc_out_1 = acc_out_internal_41;
    assign acc_out_2 = acc_out_internal_42;
    assign acc_out_3 = acc_out_internal_43;
    assign acc_out_4 = acc_out_internal_44;
    
    

endmodule

module PE #(parameter N = 8, parameter [7:0] comb = 120)(
    input wire clk,          // Clock signal
    input wire reset,        // Reset signal
    input wire control,      // Control signal (1: load weights, 0: data flow)
    input wire signed [2*N-1:0] acc_in,
    input wire [N-1:0] B1_in,  // B weight
    input wire [N-1:0] B2_in,  // 2B weight
//    input wire [N-1:0] B_in,
    input wire [(N/2) -1:0] neg_in, //control signal neg 
    input wire [(N/2) -1:0] one_in, //control signal one
    input wire [(N/2) -1:0] two_in, //control signal two
    output reg [N-1:0] B1_out,  // B weight
    output reg [N-1:0] B2_out,  // 2B weight
//    output reg [N-1:0] B_out,
    output reg [(N/2) -1:0] neg_out, //control signal neg 
    output reg [(N/2) -1:0] one_out, //control signal one
    output reg [(N/2) -1:0] two_out, //control signal two
    output reg signed [2*N-1:0] acc_out
);

//    reg [N-1:0] weight;  // Stored weight value in the PE
//    reg [N-1:0] data_reg;  // Stored data value in the PE
    reg [N-1:0] B1_reg;
    reg [N-1:0] B2_reg;
    wire [2*N-1:0] B_in;
    wire [N-1:0] acc_out_reg;
    wire [2*N-1:0] booth_product;
 
    assign B_in = {{B2_reg, B1_reg}}; 

    
    booth_recoding #(.N(N), .comb(comb)) booth (
    .neg(neg_in), 
    .two(two_in), 
    .one(one_in),
    .B(B_in),
    .prod(booth_product)

);


    // Process for handling weight loading and data flow
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            acc_out  <= 0;
            B1_out   <= 0;
            B2_out   <= 0;
//            B_out    <= 0;
            neg_out  <= 0;
            one_out  <= 0;
            two_out  <= 0;
        end else begin
            if (control) begin
                // Load and forward Booth control + multiplicand
                B1_out   <= B1_in;
                B1_reg   <= B1_in;
                B2_out   <= B2_in;
                B2_reg   <= B2_in;
//                B_out   <= B_in;
                neg_out  <= 0;
                one_out  <= 0;
                two_out  <= 0;
                acc_out  <= 0;
            end else begin
                // MAC computation
                acc_out  <= acc_in + booth_product;
                neg_out  <= neg_in;
                one_out  <= one_in;
                two_out  <= two_in;
            end
        end
    end  

  

endmodule

module booth_recoding #(parameter N = 8, parameter [7:0] comb = 120)(
    input [(N/2) -1 :0] neg, two, one,
    input [2*N - 1:0] B,
    output signed [2*N -1 :0] prod

);


wire signed [2*N - 1:0] PP0, PP1, PP2, PP3;

generate
	case(comb[1:0])
		2'b00: partial_prod_gen_exact pp0(B, neg[0:0], two[0:0], one[0:0], PP0);
		2'b01: partial_prod_gen_approx1 pp0(B, two[0:0], one[0:0], PP0);
		2'b10: partial_prod_gen_approx2 pp0(B, neg[0:0], two[0:0], one[0:0], PP0);
		2'b11: partial_prod_gen_approx3 pp0(B, two[0:0], one[0:0], PP0);
	endcase
endgenerate

generate
	case(comb[3:2])
		2'b00: partial_prod_gen_exact pp1(B, neg[1:1], two[1:1], one[1:1], PP1);
		2'b01: partial_prod_gen_approx1 pp1(B, two[1:1], one[1:1], PP1);
		2'b10: partial_prod_gen_approx2 pp1(B, neg[1:1], two[1:1], one[1:1], PP1);
		2'b11: partial_prod_gen_approx3 pp1(B, two[1:1], one[1:1], PP1);
	endcase
endgenerate

generate
	case(comb[5:4])
		2'b00: partial_prod_gen_exact pp2(B, neg[2], two[2], one[2], PP2);
		2'b01: partial_prod_gen_approx1 pp2(B, two[2], one[2], PP2);
		2'b10: partial_prod_gen_approx2 pp2(B, neg[2], two[2], one[2], PP2);
		2'b11: partial_prod_gen_approx3 pp2(B, two[2], one[2], PP2);
	endcase
endgenerate

generate
	case(comb[7:6])
		2'b00: partial_prod_gen_exact pp3(B, neg[3], two[3], one[3], PP3);
		2'b01: partial_prod_gen_approx1 pp3(B, two[3], one[3], PP3);
		2'b10: partial_prod_gen_approx2 pp3(B, neg[3], two[3], one[3], PP3);
		2'b11: partial_prod_gen_approx3 pp3(B, two[3], one[3], PP3);
	endcase
endgenerate

// partial_prod_gen_exact #(.comb(comb[1:0]) pp0(B, neg[0:0], two[0:0], one[0:0], PP0);
// partial_prod_gen #(.comb(comb[3:2]) pp1(B, neg[1:1], two[1:1], one[1:1], PP1);
// partial_prod_gen #(.comb(comb[5:4]) pp2(B, neg[2], two[2], one[2], PP2);
// partial_prod_gen #(.comb(comb[7:6]) pp3(B, neg[3], two[3], one[3], PP3);


assign prod = PP0 + (PP1<<2) + (PP2<<4) + (PP3<<6);

endmodule

module partial_prod_gen_exact (
    input signed [15:0] B,
//    input signed [7:0] B,
    input neg, two, one,
    output reg signed [15:0] PP1
);

reg signed [7:0] PP2;
wire [2:0] signal;
assign signal = {{neg, two, one}};

always@(*) begin
    PP2 = 0;
    if(one == 1) begin
        
        if(neg == 1) begin
            PP2 = ~B[7:0];
            PP1 = PP2 + 1;
        
        end
        
        else begin
        
            PP2 = {{8{{B[7]}}}};
            PP1 = {{PP2,B[7:0]}};
        end
        
    end
        
    else if (two == 1)begin
    
        if(neg == 1) begin
            PP2 = ~B[15:8];
            PP1 = PP2 + 1;
        
        end
        
        else begin
            PP2 = {{8{{B[15]}}}};
            PP1 = {{PP2, B[15:8]}};
        end
     end
        
    else begin
    
        PP1 = 16'b0;
        
    end

    
end
    
endmodule

////////////////////////////// approx 1 //////////////////////////////

module partial_prod_gen_approx1 (
    input signed [15:0] B,
//    input signed [7:0] B,
    input two, one,
    output reg signed [15:0] PP1
);

reg signed [7:0] PP2;
wire [1:0] signal;
assign signal = {{ two, one}};

always@(*) begin
    PP2 = 0;
    if(one == 1) begin
        
            PP2 = {{8{{B[7]}}}};
            PP1 = {{PP2,B[7:0]}};
        
    end
        
    else if (two == 1)begin
    
            PP2 = {{8{{B[15]}}}};
            PP1 = {{PP2, B[15:8]}};
        
     end
        
    else begin
    
        PP1 = 16'b0;
        
    end

    
end

////////////////////////////// approx 2 //////////////////////////////

endmodule

module partial_prod_gen_approx2 (
    input signed [15:0] B,
//    input signed [7:0] B,
    input neg, two, one,
    output reg signed [15:0] PP1
);

reg signed [7:0] PP2;
wire [2:0] signal;
assign signal = {{ neg, two, one}};

always@(*) begin
    PP2 = 0;
   if(one == 1 || two == 1) begin
        
        if(neg == 1) begin
            PP2 = ~B[7:0];
            PP1 = PP2 + 1;
        
        end
        
        else begin
        
            PP2 = {{8{{B[7]}}}};
            PP1 = {{PP2,B[7:0]}};
        end
        
    end
        
    else begin
    
        PP1 = 16'b0;
        
    end

    
end

endmodule

////////////////////////////// approx 3 //////////////////////////////

module partial_prod_gen_approx3 (
    input signed [15:0] B,
//    input signed [7:0] B,
    input two, one,
    output reg signed [15:0] PP1
);

reg signed [7:0] PP2;
wire [2:0] signal;
assign signal = {{neg, two, one}};

always@(*) begin
    PP2 = 0;
    if(one == 1) begin
            PP2 = ~B[7:0];
            PP1 = PP2 + 1;
                
    end
        
    else if (two == 1)begin
    
            PP2 = ~B[15:8];
            PP1 = PP2 + 1;
        
       
     end
        
    else begin
    
        PP1 = 16'b0;
        
    end

    
end

endmodule

module booth_control_top #(parameter N = 8) (

    input [N-1:0] A, B,
    output [2*N - 1 : 0] prod
    
);
    wire [N/2 -1 :0] neg, two, one;
    wire signed [N:0] ext_multiplier = {{A, 1'b0}};

    (DONT_TOUCH="YES") control_sig_gen gen1(.booth_bits(ext_multiplier[2:0]), .neg(neg[0]), .two(two[0]), .one(one[0]));
    (DONT_TOUCH="YES") control_sig_gen gen2(.booth_bits(ext_multiplier[4:2]), .neg(neg[1]), .two(two[1]), .one(one[1]));
    (DONT_TOUCH="YES") control_sig_gen gen3(.booth_bits(ext_multiplier[6:4]), .neg(neg[2]), .two(two[2]), .one(one[2]));
    (DONT_TOUCH="YES") control_sig_gen gen4(.booth_bits(ext_multiplier[8:6]), .neg(neg[3]), .two(two[3]), .one(one[3]));

//    wire [3:0] neg, two, one;
   booth_recoding booth(neg, two, one, B, prod);

endmodule

///////// Control Signal Generation //////

module control_sig_gen (
    input [2:0] booth_bits,
    output reg neg, two, one
  
);
    wire [16:0] signed_multiplicand;
    

  
    always @(*) begin
          case (booth_bits)

                3'b000, 3'b111: begin one = 0; neg = 0; two = 0; end           

                3'b001, 3'b010: begin one = 1; neg = 0; two = 0; end // No addition

                3'b011: begin one = 0; neg = 0; two = 1; end // Add A shifted by 2*i

                3'b100: begin one = 0; neg = 1; two = 1; end // Subtract 2 * A shifted

                3'b101, 3'b110:  begin neg = 1; one = 1; two = 0; end 
        endcase
       
    end
endmodule



module booth_recoding #(parameter N = 8, parameter [7:0] comb = 0)(
    input [(N/2) -1 :0] neg, two, one,
    input [N - 1:0] BB,
    output signed [2*N -1 :0] prod

);
// wire [2*N :0] B;
// assign B = {{BB<<1, BB}};

wire [8:0] B2 = {{BB[7], BB}} << 1;  // Left-shift with MSB extension
wire [16:0] B = {{B2, BB}}; // {{9-bit 2B, 8-bit B}}

wire signed [2*N -1 :0] PP0, PP1, PP2, PP3;

generate
	case(comb[1:0])
		2'b00: partial_prod_gen_exact pp0(B, neg[0:0], two[0:0], one[0:0], PP0);
		2'b01: partial_prod_gen_approx1 pp0(B, two[0:0], one[0:0], PP0);
		2'b10: partial_prod_gen_approx2 pp0(B, neg[0:0], two[0:0], one[0:0], PP0);
		2'b11: partial_prod_gen_approx3 pp0(B, two[0:0], one[0:0], PP0);
	endcase
endgenerate

generate
	case(comb[3:2])
		2'b00: partial_prod_gen_exact pp1(B, neg[1:1], two[1:1], one[1:1], PP1);
		2'b01: partial_prod_gen_approx1 pp1(B, two[1:1], one[1:1], PP1);
		2'b10: partial_prod_gen_approx2 pp1(B, neg[1:1], two[1:1], one[1:1], PP1);
		2'b11: partial_prod_gen_approx3 pp1(B, two[1:1], one[1:1], PP1);
	endcase
endgenerate

generate
	case(comb[5:4])
		2'b00: partial_prod_gen_exact pp2(B, neg[2], two[2], one[2], PP2);
		2'b01: partial_prod_gen_approx1 pp2(B, two[2], one[2], PP2);
		2'b10: partial_prod_gen_approx2 pp2(B, neg[2], two[2], one[2], PP2);
		2'b11: partial_prod_gen_approx3 pp2(B, two[2], one[2], PP2);
	endcase
endgenerate

generate
	case(comb[7:6])
		2'b00: partial_prod_gen_exact pp3(B, neg[3], two[3], one[3], PP3);
		2'b01: partial_prod_gen_approx1 pp3(B, two[3], one[3], PP3);
		2'b10: partial_prod_gen_approx2 pp3(B, neg[3], two[3], one[3], PP3);
		2'b11: partial_prod_gen_approx3 pp3(B, two[3], one[3], PP3);
	endcase
endgenerate


// Compressors
 	wire fs[15:0], fc[15:0] , Pc[15:0], fc_out[15:0];
    //wire Pf [15:0];
    wire [0:0] P [15:0];
    
    assign P[0]  = PP0[0:0];

	assign P[1]  = PP0[1];

    assign {{ fc[2], P[2] }} = PP0[2:2] + PP1[0:0] ;

    assign {{ fc[3], fs[3] }} = PP0[3:3] + PP1[1:1];
    
    compres comp04(PP0[4:4],PP1[2:2], PP2[0:0], 1'b0, 1'b0, fs[4],fc_out[4], fc[4]);

    compres comp05(PP0[5:5],PP1[3:3], PP2[1:1], fc_out[4], 1'b0, fs[5],fc_out[5], fc[5]);
  
    compres comp06(PP0[6:6],PP1[4:4], PP2[2:2], PP3[0:0], fc_out[5], fs[6],fc_out[6], fc[6]);

    compres comp07(PP0[7:7],PP1[5:5], PP2[3:3], PP3[1:1], fc_out[6], fs[7],fc_out[7], fc[7]);

    compres comp08(PP0[8:8],PP1[6:6], PP2[4:4], PP3[2:2], fc_out[7], fs[8],fc_out[8], fc[8]);

    compres comp09(PP0[9:9],PP1[7:7], PP2[5:5], PP3[3:3], fc_out[8], fs[9],fc_out[9], fc[9]);

	compres comp10(PP0[10:10], PP1[8:8],  PP2[6:6], PP3[4:4], fc_out[9],  fs[10], fc_out[10], fc[10]);
    compres comp11(PP0[11:11], PP1[9:9],  PP2[7:7], PP3[5:5], fc_out[10], fs[11], fc_out[11], fc[11]);
    compres comp12(PP0[12:12], PP1[10:10], PP2[8:8], PP3[6:6], fc_out[11], fs[12], fc_out[12], fc[12]);
    compres comp13(PP0[13:13], PP1[11:11], PP2[9:9], PP3[7:7], fc_out[12], fs[13], fc_out[13], fc[13]);
    compres comp14(PP0[14:14], PP1[12:12], PP2[10:10], PP3[8:8], fc_out[13], fs[14], fc_out[14], fc[14]);
    compres comp15(PP0[15:15], PP1[13:13], PP2[11:11], PP3[9:9], fc_out[14], fs[15], fc_out[15], fc[15]);

	//2nd level
	

	FullAdder fa3(fs[3], fc[2], 1'b0, Pc[3], P[3]);

    FullAdder fa4(fs[4], fc[3], Pc[3], Pc[4], P[4]);

    FullAdder fa5(fs[5], fc[4], Pc[4], Pc[5], P[5]);

    FullAdder fa6(fs[6], fc[5], Pc[5], Pc[6], P[6]);

    FullAdder fa7(fs[7], fc[6], Pc[6], Pc[7], P[7]);

    FullAdder fa8(fs[8], fc[7], Pc[7], Pc[8], P[8]);

    FullAdder fa9(fs[9], fc[8], Pc[8], Pc[9], P[9]);

    FullAdder fa10(fs[10], fc[9], Pc[9], Pc[10], P[10]);

    FullAdder fa11(fs[11], fc[10],Pc[10], Pc[11], P[11]);

    FullAdder fa12(fs[12], fc[11], Pc[11], Pc[12], P[12]);

    FullAdder fa13(fs[13], fc[12], Pc[12], Pc[13], P[13]);

    FullAdder fa84(fs[14], fc[13], Pc[13], Pc[14], P[14]);

    FullAdder fa15(fs[15], fc[14], Pc[14], Pc[15], P[15]);



assign prod = {{P[15],P[14],P[13], P[12], P[11], P[10], P[9], P[8], P[7], P[6], P[5], P[4], P[3], P[2], P[1], P[0]}};

endmodule

module partial_prod_gen_exact (
    input signed [16:0] B,
//    input signed [7:0] B,
    input neg, two, one,
    output reg signed [15:0] PP1
);

reg signed [8:0] PP2;
reg signed [9:0] PP2_2;
// reg signed [15:0] PP1;
wire [2:0] signal;
assign signal = {{neg, two, one}};
reg sign;

always@(*) begin
    PP2 = 0;
    if(one == 1) begin
        
        if(neg == 1) begin
            PP2 = {{B[7], B[7:0]}};
            PP2 = (~PP2) + 1'b1;
            PP1 = {{{{7{{PP2[8]}}}},PP2}};
        
        end
        
        else begin
        
            PP2 = {{8{{B[7]}}}};
            PP1 = {{PP2,B[7:0]}};
        end
        
    end
        
    else if (two == 1)begin
    
        if(neg == 1) begin
            PP2_2 = {{B[16],B[16:8]}};
            
            PP2_2 = (~PP2_2) + 1'b1;
            PP1 = {{{{7{{PP2_2[9]}}}},PP2_2}};
        
        end
        
        else begin
            PP2_2 = {{7{{B[16]}}}};
            PP1 = {{PP2_2[6:0], B[16:8]}};
        end
     end
        
    else begin
    
        PP1 = 16'b0;
        
    end

end

  
endmodule


module compres ( 

    input A, B, C, D , 

    input C_IN , 

    output SUM, 

    output C_OUT_ext , 

    output C_OUT_int ); 

	wire internal_sum ; 

	FullAdder fa_upper ( .a(A) , .b(B) , .cin(C) , .cout(C_OUT_ext) , .sum(internal_sum) ) ; 

	FullAdder fa_lower ( .a(C_IN) , .b(internal_sum) , .cin(D) , .cout(C_OUT_int) , .sum(SUM) ) ; 

	

endmodule 

module FullAdder ( 

	input a, b, cin , 
    output cout , sum ) ; 

    assign {{ cout , sum }} = a + b + cin ; 

endmodule


"""
    with open("systolic_array.v", "w") as f:
        f.write(verilog)
        print("Verilog generation completed.")

#----------------------- Run Genus --------------------------# 

def run_genus():
    with open("run_genus.tcl", "w") as f:
        f.write("""

set tech_node "45nm"

# Cell Type
set cell_type "fast_vdd1v0_basicCells.lib"  

# HDL file
set hdl_file "systolic_array"

# Synthesis efforts
set generic_effort "medium"
set map_effort "medium"
set opt_effort "medium"
                
set DATE [clock format [clock seconds] -format "%b%d-%T"] 

set_db hdl_unconnected_value 0 
#set_db timing_report_unconstrained true

# Directory to search for .lib (Library) files
set_db init_lib_search_path ./pdks/$tech_node/lib/

# Directory to search RTL files
set_db init_hdl_search_path ./rtl/$hdl_file/
#set_db init_hdl_search_path ./

# unflatten: individual optimization
set_db auto_ungroup none

# Verbose info level 0-9 (Recommended-6, Max-9)
set_db information_level 7

# Write log file for each run
set_db stdout_log ./genus/logs/${hdl_file}_log.txt

# Stop genus from executing when it encounters error
set_db fail_on_error_mesg true

# This attribute enables Genus to keep track of filenames, line numbers, and column numbers
# for all instances before optimization. Genus also uses this information in subsequent error
# and warning messages.
set_db hdl_track_filename_row_col true


# This is for timing optimization read genus legacy UI documentation if results are worst
#set_db tns_opto true /

# Choose the lib cell type
set_db library $cell_type 


# Naming style used in rtl
set_db hdl_parameter_naming_style _%s%d 

# Automatically partition the design and run fast in genus
set_db auto_partition true

# Check DRC & force Genus to fix DRCs, even at the expense of timing, with the drc_first attribute.
set_db drc_first true 

# Read verilog file ( if it is sv just replace the extension)
# read_hdl -mixvlog ./rtl/$hdl_file/$hdl_file.v
read_hdl -mixvlog ./$hdl_file.v

set top_module $hdl_file

# Elaborate the design
elaborate

# Check for unresolved refernces # Technology independent
check_design -unresolved > ./genus/reports/$hdl_file/$tech_node/design_check.rpt

# Read the constraint file # Technology Independent
read_sdc ./constraints/$hdl_file/$hdl_file.sdc


# Generic Synthesis
set_db syn_generic_effort $generic_effort
syn_generic 
write_hdl > ./genus/synthesis/$hdl_file/$tech_node/generic/${hdl_file}_syn_generic.v
write_sdc > ./genus/synthesis/$hdl_file/$tech_node/generic/${hdl_file}_syn_generic.sdc
report_power > ./genus/reports/$hdl_file/$tech_node/generic/${hdl_file}_syn_generic_power.rpt
write_snapshot -outdir ./genus/reports/$hdl_file/$tech_node/generic/ -tag ${hdl_file}_syn_generic
report_power > ./genus/reports/$hdl_file/$tech_node/generic/${hdl_file}_syn_generic_power.rpt

#set_dont_touch [get_designs systolic_array] true
#set_dont_touch [get_cells -hierarchical pe_*] 
#set_dont_touch [get_cells -hierarchical pe_*/multi*]
#set_dont_touch [get_cells -hierarchical pe_*/add*]

# Mapping 
set_db syn_map_effort $map_effort
syn_map
time_info MAPPED
write_hdl > ./genus/synthesis/$hdl_file/$tech_node/mapped/${hdl_file}_syn_map.v
write_sdc > ./genus/synthesis/$hdl_file/$tech_node/mapped/${hdl_file}_syn_map.sdc
report_power > ./genus/reports/$hdl_file/$tech_node/mapped/${hdl_file}_syn_map_power.rpt
write_snapshot -outdir ./genus/reports/$hdl_file/$tech_node/mapped/ -tag ${hdl_file}_syn_map
report_power > ./genus/reports/$hdl_file/$tech_node/mapped/${hdl_file}_syn_map_power.rpt

# step 1 LEC do file generation
write_hdl -lec > ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_pre_opt.v
write_do_lec -golden_design rtl -revised_design ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_pre_opt.v > ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_pre_opt.do

set_dont_touch [get_cells -hierarchical control*/gen*]
set_dont_touch [get_cells -hierarchical pe_*]
set_dont_touch [get_cells -hierarchical pe_*/booth]
set_dont_touch [get_cells -hierarchical pe*/booth/gen*]

# Incremental performs area and power optimization

# Optimized
set_db syn_opt_effort $opt_effort
syn_opt -incr
time_info OPT
write_hdl > ./genus/synthesis/$hdl_file/$tech_node/opt/${hdl_file}_syn_opt.v
write_sdc > ./genus/synthesis/$hdl_file/$tech_node/opt/${hdl_file}_syn_opt.sdc
report_power > ./genus/reports/$hdl_file/$tech_node/opt/${hdl_file}_syn_opt_power.rpt
write_snapshot -outdir ./genus/reports/$hdl_file/$tech_node/opt/ -tag ${hdl_file}_syn_opt
report_summary -directory ./genus/reports/$hdl_file/$tech_node/
report_power > ./genus/reports/$hdl_file/$tech_node/opt/${hdl_file}_syn_opt_power.rpt


# step 2 LEC do file generation for synthesized netlist
write_hdl -lec > ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_opt.v
write_do_lec -golden_design ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_pre_opt.v -revised_design ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_opt.v > ./genus/lec/$hdl_file/$tech_node/${hdl_file}_lec_opt.do

# Report design rules
report_design_rules > ./genus/reports/$hdl_file/$tech_node/des-Rules.rpt

gui_show


""")
    
    subprocess.run(["genus", "-batch", "-f", "run_genus.tcl"], check=True)
    print("Genus synthesis completed.")

#----------------------- Parse Thrugh Reports --------------------------# 

def parse_area():
    with open("./genus/reports/systolic_array/45nm/opt/systolic_array_syn_opt_qor.rpt", "r") as file:
        for line in file:
            if "Total Area (Cell+Physical+Net)" in line:
                parts = line.split()
                try:
                    return float(parts[-1])
                except ValueError:
                    return None

def parse_power():
    with open("./genus/reports/systolic_array/45nm/opt/systolic_array_syn_opt_power.rpt", "r") as file:
        for line in file:
            if "Subtotal" in line:
                parts = line.split()
                try:
                    return float(parts[-2])
                except ValueError:
                    return None
    return None

def parse_timing():
    with open("./genus/reports/systolic_array/45nm/opt/systolic_array_syn_opt_time.rpt", "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "Data Path:-" in line:
            try:
                # Extract the floating-point value at the end of the line
                return float(line.strip().split()[-1])
            except ValueError:
                return None
            break  # Exit after finding the first Data Path

    return None

#----------------------- PE Configuration Evaluation --------------------------#
def evaluate_pe_configuration(pe_config_list, sample_limit=None):
    """
    Evaluate a PE configuration and return accuracy, power, area metrics
    """
    if sample_limit is None:
        sample_limit = EVAL_SAMPLES
    
    pe_config_obj = PEConfig(pe_config_list)
    correct = 0
    total = 0

    total_operations = 0
    exact_operations = 0
    
    pe_index = 0

    generate_verilog_from_combs(pe_config_list)

    run_genus()

    area_score = parse_area() or 1.0  # fallback if parsing fails
    power_score = parse_power() or 1.0
    delay_score = parse_timing() or 0.0
    
    with torch.no_grad():
        for i, (image, label) in enumerate(global_dataloader):
            if i >= sample_limit:
                break
                
            label = label.item()
            
            # Fast input quantization
            img_np = image[0].squeeze().numpy()
            x = ((img_np * 0.3081 + 0.1307) * 255 - 128).astype(np.int8)
            x = np.clip(x, -128, 127)

            # Conv1 - Use PE configuration
            conv1_features = []
            for ch in range(6):
                kernel = global_weights['conv1_w_q'][ch, 0]
                pe_variants = pe_config_obj.get_pe_variants(pe_index % pe_config_obj.num_pes)
                pe_index += 1
                conv_out = vectorized_booth_conv_with_pe(x, kernel, pe_variants, padding=2)
                conv_out = (conv_out.astype(np.float32) * global_weights['conv1_scale'] + global_weights['conv1_b'][ch])
                conv_out = np.maximum(conv_out, 0).astype(np.int32)
                pooled = max_pool_2x2(conv_out)
                conv1_features.append(pooled)
            
            conv1_out = np.stack(conv1_features)

            # Conv2 - Use PE configuration
            conv2_features = []
            for ch in range(16):
                acc = np.zeros((12, 12), dtype=np.int32)
                for in_ch in range(6):
                    kernel = global_weights['conv2_w_q'][ch, in_ch]
                    pe_variants = pe_config_obj.get_pe_variants(pe_index % pe_config_obj.num_pes)
                    pe_index += 1
                    partial = vectorized_booth_conv_with_pe(conv1_out[in_ch], kernel, pe_variants, padding=0)
                    acc += partial
                
                acc = (acc.astype(np.float32) * global_weights['conv2_scale'] + global_weights['conv2_b'][ch])
                acc = np.maximum(acc, 0).astype(np.int32)
                pooled = max_pool_2x2(acc)
                conv2_features.append(pooled)
            
            conv2_out = np.stack(conv2_features)

            # FC layers
            x_tensor = torch.tensor(conv2_out, dtype=torch.float32).unsqueeze(0)
            logits = global_model.classifier(x_tensor.view(1, -1))
            pred = logits.argmax(1).item()

            if pred == label:
                correct += 1
            total += 1
    

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, 0, 0, 0


#----------------------- NSGA-II Optimization Framework --------------------------#

class CustomMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for j in range(len(X[i])):
                if random.random() < MUTATION_RATE:
                    X[i][j] = random.randint(int(problem.xl[j]), int(problem.xu[j]))
        return X

class MyCallback:
    def __init__(self):
        self.data = []
        self.generation = 0

    def __call__(self, algorithm):
        F = algorithm.pop.get('F')
        self.data.append(F)
        
        # Print progress
        if self.generation % 5 == 0:
            best_acc = np.max(F[:, 0])  # Assuming accuracy is first objective
            avg_acc = np.mean(F[:, 0])
            print(f"Gen {self.generation}: Best Acc: {best_acc:.4f}, Avg Acc: {avg_acc:.4f}")
        
        self.generation += 1

class PEConfigProblem(Problem):
    def __init__(self, **kwargs):
        # Each PE has 4 partial products, each can be 0-3 (4 variants)
        # We have NUM_PES processing elements
        n_var = NUM_PES * 4
        xl = [0] * n_var  # Lower bounds
        xu = [3] * n_var  # Upper bounds (variants 0, 1, 2, 3)
        
        super().__init__(
            n_var=n_var, 
            n_obj=4,  # Accuracy (maximize), Power (minimize), Area (minimize), Delay(minimize)
            n_ieq_constr=0,
            xl=xl, 
            xu=xu,
            vtype=int,
            elementwise_evaluation=False,
            **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Parallel evaluation of solutions
        params = [[X[k], k] for k in range(len(X))]
        results = pool.starmap(self.evaluate_single_solution, params)
        
        F = []
        for result in results:
            accuracy, power, area, delay = result
            # Convert to minimization problem (NSGA-II minimizes)
            F.append([
                -accuracy,  # Maximize accuracy -> minimize negative accuracy
                power,      # Minimize power
                area ,       # Minimize area
                delay
            ])
        
        out["F"] = np.array(F)

    def evaluate_single_solution(self, x, solution_id):
        """Evaluate a single PE configuration"""
        # Convert flat array to list of tuples
        pe_config = []
        for i in range(0, len(x), 4):
            pe_config.append(tuple(x[i:i+4]))
        
        # Evaluate the configuration
        accuracy, power, area, delay = evaluate_pe_configuration(pe_config, sample_limit=EVAL_SAMPLES)
        
        return accuracy, power, area, delay

def determineDecisionVariableLimit():
    """Determine decision variable limits"""
    n_var = NUM_PES * 4
    xl = [0] * n_var
    xu = [3] * n_var
    return xl, xu

def runFramework():
    """Main optimization framework"""
    print(" Starting PE Configuration Optimization Framework")
    print(f"Population: {POPULATION}, Generations: {GENERATIONS}")
    print(f"Number of PEs: {NUM_PES}, Evaluation Samples: {EVAL_SAMPLES}")
    
    # Setup global data
    setup_global_data()
    
    # Warm up JIT compiler
    print("Warming up JIT compiler...")
    dummy_config = [(0, 1, 2, 3) for _ in range(NUM_PES)]
    _ = evaluate_pe_configuration(dummy_config, sample_limit=10)
    print(" JIT compiler warmed up")
    
    # Setup optimization
    problem = PEConfigProblem()
    callback = MyCallback()
    algorithm = NSGA2(
        pop_size=POPULATION,
        sampling=IntegerRandomSampling(),
        crossover=PointCrossover(n_points=2, prob=0.8),
        mutation=CustomMutation()
    )
    termination = get_termination("n_gen", GENERATIONS)
    
    print(" Starting NSGA-II optimization...")
    start_time = time.time()
    
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=False,
        callback=callback,
        seed=SEED,
        verbose=False
    )
    
    end_time = time.time()
    print(f" Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Process results
    pool.close()
    
    print("\n Final Results:")
    objectives = np.array(res.F)
    solutions = np.array(res.X.astype(int))
    
    # Convert back to maximization for accuracy
    objectives[:, 0] = -objectives[:, 0]
    
    print("Pareto Front Solutions:")
    print("Accuracy | Power | Area | Delay")
    print("-" * 25)
    
    # Sort by accuracy (descending)
    sorted_indices = np.argsort(-objectives[:, 0])
    
    best_solutions = []
    for i in sorted_indices[:5]:  # Top 5 solutions
        acc, power, area, delay = objectives[i]
        solution = solutions[i]
        
        # Convert solution back to PE configuration
        pe_config = []
        for j in range(0, len(solution), 4):
            pe_config.append(tuple(solution[j:j+4]))
        
        print(f"{acc:.4f}   | {power:.4f} | {area:.4f}")
        best_solutions.append((acc, power, area, delay, pe_config))
    
    # Save best solution
    best_acc, best_power, best_area, best_delay, best_pe_config = best_solutions[0]
    
    print(f"\n Best Solution:")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Power Score: {best_power:.4f}")
    print(f"Area Score: {best_area:.4f}")
    print(f"Delay Score: {best_delay:.4f}")
    print(f"PE Configuration: {best_pe_config[:3]}...")  # Show first 3 PEs
    
    # Save results to file
    with open("pe_optimization_results.txt", "w") as f:
        f.write("PE Configuration Optimization Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Population: {POPULATION}\n")
        f.write(f"Generations: {GENERATIONS}\n")
        f.write(f"Number of PEs: {NUM_PES}\n")
        f.write(f"Evaluation Samples: {EVAL_SAMPLES}\n")
        f.write(f"Optimization Time: {end_time - start_time:.2f} seconds\n\n")
        
        f.write("Best Solutions:\n")
        for i, (acc, power, area, delay, pe_config) in enumerate(best_solutions):
            f.write(f"Solution {i+1}: Acc={acc:.4f}, Power={power:.4f}, Area={area:.4f}, Delay={delay:.4f}\n")
            f.write(f"PE Config: {pe_config}\n\n")
    
    return best_pe_config, objectives, solutions

    

if __name__ == '__main__':
    print("PE Configuration Optimization Framework")
    print("=" * 50)
    
    try:
        best_config, all_objectives, all_solutions = runFramework()
        print(" Optimization completed successfully!")
        print(" Results saved to 'pe_optimization_results.txt'")
    except Exception as e:
        print(f" Error during optimization: {e}")
        import traceback
        traceback.print_exc()
