`timescale 1ns / 1ps

module uart_tx #(
    parameter integer CLK_FREQ  = 125_000_000,
    parameter integer BAUD_RATE = 115200
)(
    input  wire      clk,
    input  wire      send,
    input  wire [7:0] data,

    output reg       tx,
    output reg       busy
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    reg [31:0] clk_count;
    reg [3:0]  bit_index;
    reg [9:0]  tx_shift;

    always @(posedge clk) begin

        if (!busy) begin

            tx <= 1'b1;

            if (send) begin

                // UART:
                // START + 8 DATA + STOP
                tx_shift <= {1'b1, data, 1'b0};

                busy      <= 1'b1;
                bit_index <= 0;
                clk_count <= 0;

                tx <= 1'b0;
            end

        end
        else begin

            if (clk_count < CLKS_PER_BIT - 1) begin

                clk_count <= clk_count + 1;

            end
            else begin

                clk_count <= 0;

                if (bit_index < 9) begin

                    bit_index <= bit_index + 1;
                    tx <= tx_shift[bit_index + 1];

                end
                else begin

                    busy      <= 1'b0;
                    tx        <= 1'b1;
                    bit_index <= 0;

                end
            end
        end
    end

    initial begin
        tx        = 1'b1;
        busy      = 1'b0;
        clk_count = 0;
        bit_index = 0;
        tx_shift  = 10'b1111111111;
    end

endmodule