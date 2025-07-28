
module systolic_array #(
    parameter N = 8,
    parameter [7:0] comb0 = 0,
    parameter [7:0] comb1 = 0,
    parameter [7:0] comb2 = 0,
    parameter [7:0] comb3 = 0,
    parameter [7:0] comb4 = 0,
    parameter [7:0] comb5 = 0,
    parameter [7:0] comb6 = 0,
    parameter [7:0] comb7 = 0,
    parameter [7:0] comb8 = 0,
    parameter [7:0] comb9 = 0,
    parameter [7:0] comb10 = 0,
    parameter [7:0] comb11 = 0,
    parameter [7:0] comb12 = 0,
    parameter [7:0] comb13 = 0,
    parameter [7:0] comb14 = 0,
    parameter [7:0] comb15 = 0,
    parameter [7:0] comb16 = 0,
    parameter [7:0] comb17 = 0,
    parameter [7:0] comb18 = 0,
    parameter [7:0] comb19 = 0,
    parameter [7:0] comb20 = 0,
    parameter [7:0] comb21 = 0,
    parameter [7:0] comb22 = 0,
    parameter [7:0] comb23 = 0,
    parameter [7:0] comb24 = 0
            

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

