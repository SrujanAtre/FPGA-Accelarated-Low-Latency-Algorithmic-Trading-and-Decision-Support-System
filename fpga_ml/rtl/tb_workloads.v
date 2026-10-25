`timescale 1ns/1ps
`include "model_config.vh"
module tb_workloads;
 reg clk=0;always #4 clk=~clk;
 reg reset=1,start=0;reg [863:0] features=0;
 wire busy,done;wire [1:0] prediction;wire [31:0] cycles;wire signed [63:0] s0,s1,s2;
 ml_core dut(clk,reset,start,features,busy,done,prediction,cycles,s0,s1,s2);
 reg [31:0] vectors[0:`N_SAMPLES*27-1];reg [1:0] expected[0:`N_SAMPLES-1];
 integer tick=0;always @(posedge clk)tick<=tick+1;
 integer mode,i,j,f,period,base,arrival,accepted,complete;
 initial begin
  $readmemh("vectors.mem",vectors);$readmemh("expected.mem",expected);
  f=$fopen("workload_sim_results.csv","w");$fwrite(f,"period_cycles,sample,arrival_cycle,accepted_cycle,completion_cycle,latency_cycles,queue_delay_cycles\n");
  for(mode=0;mode<3;mode=mode+1) begin
   reset=1;repeat(5)@(negedge clk);reset=0;
   period=mode==0?2000:(mode==1?1000:500);base=tick+2;
   for(i=0;i<`N_SAMPLES;i=i+1) begin
    arrival=base+i*period;
    while(tick<arrival)@(negedge clk);
    for(j=0;j<27;j=j+1)features[j*32+:32]=vectors[i*27+j];
    start=1;accepted=tick;@(negedge clk);start=0;
    wait(done);@(negedge clk);complete=tick;
    if(prediction!==expected[i])$fatal(1,"Workload mismatch");
    $fwrite(f,"%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",period,i,arrival,accepted,complete,cycles,accepted-arrival);
    @(negedge clk);
   end
  end
  $fclose(f);$display("PASS 1929 workload inferences");$finish;
 end
 initial begin #100000000;$fatal(1,"Timeout");end
endmodule
