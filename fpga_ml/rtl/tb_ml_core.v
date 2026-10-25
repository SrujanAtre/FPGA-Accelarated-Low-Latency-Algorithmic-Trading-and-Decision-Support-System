`timescale 1ns/1ps
`include "model_config.vh"
module tb_ml_core;
 reg clk=0; always #4 clk=~clk;
 reg reset=1,start=0;reg [863:0] features=0;
 wire busy,done;wire [1:0] prediction;wire [31:0] cycles;
 wire signed [63:0] s0,s1,s2;
 ml_core dut(clk,reset,start,features,busy,done,prediction,cycles,s0,s1,s2);
 reg [31:0] vectors[0:`N_SAMPLES*27-1];reg [1:0] expected[0:`N_SAMPLES-1];
 integer i,j,f;
 initial begin
  $readmemh("vectors.mem",vectors);$readmemh("expected.mem",expected);
  f=$fopen("rtl_results.csv","w");$fwrite(f,"sample,prediction,cycles,score0,score1,score2\n");
  repeat(5) @(negedge clk); reset=0;
  for(i=0;i<`N_SAMPLES;i=i+1) begin
   for(j=0;j<27;j=j+1) features[j*32+:32]=vectors[i*27+j];
   @(negedge clk);start=1;@(negedge clk);start=0;
   wait(done);@(negedge clk);
   if(prediction!==expected[i]) $fatal(1,"Prediction mismatch sample %0d",i);
   $fwrite(f,"%0d,%0d,%0d,%0d,%0d,%0d\n",i,prediction,cycles,s0,s1,s2);
  end
  $fclose(f);$display("PASS: %0d samples",`N_SAMPLES);$finish;
 end
 initial begin #20000000;$fatal(1,"Timeout");end
endmodule
