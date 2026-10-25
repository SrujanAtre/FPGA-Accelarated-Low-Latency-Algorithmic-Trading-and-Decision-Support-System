`timescale 1ns/1ps
module fpga_ml_top(input wire CLK_125_P,CLK_125_N,uart_rx, output wire uart_tx);
 wire raw_clk,clk;
 IBUFDS #(.DIFF_TERM("FALSE"),.IBUF_LOW_PWR("TRUE")) ib(.I(CLK_125_P),.IB(CLK_125_N),.O(raw_clk));
 BUFG bg(.I(raw_clk),.O(clk));
 reg [5:0] reset_count=0;
 wire reset= !reset_count[5];
 always @(posedge clk) if(reset) reset_count<=reset_count+1;
 wire [7:0] rx_data;wire rx_valid;
 uart_rx rx(.clk(clk),.rx(uart_rx),.data(rx_data),.data_valid(rx_valid));
 reg [863:0] features;reg [7:0] index;reg receiving,start;
 wire busy,done;wire [1:0] prediction;wire [31:0] cycles;wire signed [63:0] s0,s1,s2;
 ml_core core(clk,reset,start,features,busy,done,prediction,cycles,s0,s1,s2);
 reg [2:0] tx_state;reg [5:0] tx_index;reg [239:0] response;
 reg tx_send;reg [7:0] tx_data;wire tx_busy;
 uart_tx tx(.clk(clk),.send(tx_send),.data(tx_data),.tx(uart_tx),.busy(tx_busy));
 always @(posedge clk) begin
  if(reset) begin features<=0;index<=0;receiving<=0;start<=0;tx_state<=0;tx_index<=0;tx_send<=0;tx_data<=0;response<=0;end
  else begin
   start<=0;tx_send<=0;
   if(rx_valid && !busy && tx_state==0) begin
    if(!receiving) begin if(rx_data==8'hA5) begin receiving<=1;index<=0;end end
    else begin
     // Lossless ordered float32 keys, each transmitted little endian.
     features[index*8+:8]<=rx_data;
     if(index==107) begin receiving<=0;start<=1;end else index<=index+1;
    end
   end
   if(done) begin response<={s2,s1,s0,cycles,6'b0,prediction,8'h5A};tx_index<=0;tx_state<=1;end
   case(tx_state)
    1:if(!tx_busy) begin tx_data<=response[tx_index*8+:8];tx_send<=1;tx_state<=2;end
    2:if(tx_busy) tx_state<=3;
    3:if(!tx_busy) begin if(tx_index==29) tx_state<=0;else begin tx_index<=tx_index+1;tx_state<=1;end end
   endcase
  end
 end
endmodule
