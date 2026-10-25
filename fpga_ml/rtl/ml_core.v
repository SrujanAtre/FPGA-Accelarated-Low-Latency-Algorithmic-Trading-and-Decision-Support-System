`timescale 1ns/1ps
`include "model_config.vh"
module ml_core(input wire clk, reset, start,
 input wire [863:0] features, output reg busy, done,
 output reg [1:0] prediction, output reg [31:0] cycles,
 output reg signed [63:0] score0,score1,score2);
 reg [63:0] nodes[0:`N_NODES-1];
 reg [12:0] roots[0:`N_TREES-1];
 initial begin $readmemh("nodes.mem",nodes); $readmemh("roots.mem",roots); end
 reg [12:0] addr; reg [63:0] node;
 reg [7:0] tree; reg [1:0] cls; reg [2:0] state;
 reg [863:0] latched;
 wire signed [31:0] feature = latched[node[62:58]*32 +:32];
 wire signed [31:0] value = node[31:0];
 always @(posedge clk) node <= nodes[addr];
 always @(posedge clk) begin
  if(reset) begin state<=0; busy<=0; done<=0; cycles<=0; prediction<=1; addr<=0; tree<=0; cls<=0; score0<=0;score1<=0;score2<=0;latched<=0; end
  else begin
   done<=0;
   if(busy) cycles<=cycles+1;
   case(state)
    0: if(start) begin latched<=features;busy<=1;cycles<=0;tree<=0;cls<=0;score0<=0;score1<=0;score2<=0;addr<=roots[0];state<=1;end
    1: state<=2;
    2: begin
     if(node[63]) begin
      case(cls) 0:score0<=score0+value;1:score1<=score1+value;2:score2<=score2+value;endcase
      if(tree==`N_TREES-1) state<=3;
      else begin tree<=tree+1;cls<=cls==2?0:cls+1;addr<=roots[tree+1];state<=1;end
     end else begin addr<=feature<=value?node[57:45]:node[44:32];state<=1;end
    end
    3: begin prediction<=(score0>=score1 && score0>=score2)?0:((score1>=score2)?1:2);busy<=0;done<=1;state<=0;end
   endcase
  end
 end
endmodule
