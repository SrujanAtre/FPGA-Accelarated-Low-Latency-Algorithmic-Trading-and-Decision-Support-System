`timescale 1ns/1ps
`include "model_config.vh"
module tb_uart;
 reg clk=0;always #4 clk=~clk;reg rx=1;wire tx;
 fpga_ml_top dut(clk,~clk,rx,tx);
 reg [31:0] vectors[0:`N_SAMPLES*27-1];reg [1:0] expected[0:`N_SAMPLES-1];
 reg [7:0] response[0:29]; integer i,j,k,sample,f;reg [7:0] byte_value;
 time transaction_begin,packet_finish,decision_time,transaction_end;
 always @(posedge clk) begin
  if(dut.rx_valid && dut.receiving && dut.index==107)packet_finish=$time;
  if(dut.done)decision_time=$time;
 end
 localparam BIT_NS=8680;
 task send_byte(input [7:0] value);
 integer b;
 begin rx=0;#BIT_NS;for(b=0;b<8;b=b+1)begin rx=value[b];#BIT_NS;end rx=1;#BIT_NS;end
 endtask
 initial begin
  $readmemh("vectors.mem",vectors);$readmemh("expected.mem",expected);
  f=$fopen("uart_sim_results.csv","w");$fwrite(f,"sample,prediction,cycles,input_transfer_ns,fpga_processing_ns,output_ns,total_ns\n");
  #1000;
  for(k=0;k<3;k=k+1) begin
   sample=0;while(expected[sample]!=k && sample<`N_SAMPLES-1)sample=sample+1;
   transaction_begin=$time;
   fork
    begin
     send_byte(8'hA5);
     for(i=0;i<108;i=i+1)send_byte(vectors[sample*27+i/4] >> ((i%4)*8));
    end
    begin
     for(j=0;j<30;j=j+1) begin
      @(negedge tx); #(BIT_NS+BIT_NS/2);
      for(integer b=0;b<8;b=b+1)begin response[j][b]=tx;#BIT_NS;end
      if(tx!==1'b1)$fatal(1,"UART stop bit error");
     end
    end
   join
   if(response[0]!==8'h5a || response[1]!==k)$fatal(1,"UART class/header mismatch");
   if({response[13],response[12],response[11],response[10],response[9],response[8],response[7],response[6]}!==dut.s0)$fatal(1,"UART score0 mismatch");
   #(BIT_NS/2);transaction_end=$time;
   $fwrite(f,"%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",sample,response[1],{response[5],response[4],response[3],response[2]},packet_finish-transaction_begin,decision_time-packet_finish,transaction_end-decision_time,transaction_end-transaction_begin);
   #100000;
  end
  $fclose(f);$display("PASS UART BUY HOLD SELL");$finish;
 end
 initial begin #100000000;$fatal(1,"UART timeout");end
endmodule
