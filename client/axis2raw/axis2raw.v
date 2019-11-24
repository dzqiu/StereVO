`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: NQ
// 
// Create Date: 2018/10/06 16:16:36
// Design Name: 
// Module Name: Brief_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`define VIVADO

module axis2raw #(
    parameter ADDR_WIDTH = 13, //12,//memory depth 4096     
    parameter Num_start = 8000, //4000,                       
	//这个参数为经验值，是指预存的像素值个数。因为输入的像素数据是不连续的，为了让他连续，先将像素存入fifo，然后再一起释放。
    parameter Data_WIDTH = 8
)
(
    input clk,
    input rst,
    output [Data_WIDTH-1:0] raw_data,
    output raw_valid,
    output reg [19:0] pcnt,

    `ifdef VIVADO
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 SforRaw TREADY" *)
    output wire s_axis_tready, //NOTE:the recready is not used in this current design actually, so the axis slave interface behind this design must guarntee that the ready is 1. @NQ
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 SforRaw TDATA" *)
    input wire [Data_WIDTH-1:0]s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 SforRaw TLAST" *)
    input wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 SforRaw TVALID" *)
    input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 SforRaw TUSER" *)
    input wire s_axis_tuser
    
    
    `else
    input wire [Data_WIDTH-1:0]s_axis_tdata,
    input wire s_axis_tlast,
    input wire s_axis_tuser,
    input wire s_axis_tvalid,
    output wire s_axis_tready
    `endif


);
wire valid_temp;
reg raw_ready;
reg [7:0] state;
reg [ADDR_WIDTH-1:0] cnt;
localparam FEED = 0,
           READ = 1;

wire cnt_en;
assign cnt_en = s_axis_tvalid;
always @(posedge clk)
    if(rst) begin
        state <= FEED;
        cnt <= 0;
    end
    else begin
        case(state)
            FEED: begin
                raw_ready <= 0;
                cnt <= cnt_en ? (cnt+1) : cnt;
                if(cnt == Num_start) 
                    state <= READ;
            end
            READ: begin
                raw_ready <= 1;
                cnt <= 0;
                if(!valid_temp)
                    state <= FEED;
            end
        endcase
    end
//pcnt是干嘛用的？
always @(posedge clk)
    if(rst) begin
        pcnt <= 0;
    end
    else begin
        if(raw_valid)
            pcnt <= pcnt + 1;
        if(s_axis_tuser)
            pcnt <= 0;
    end


axis_fifo #
(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(Data_WIDTH),
    .LAST_ENABLE(0),
    .ID_ENABLE(0),
    .ID_WIDTH(8),
    .DEST_ENABLE(0),
    .DEST_WIDTH(8),
    .USER_ENABLE(0),
    .USER_WIDTH(1)
)
axis_fifo_U1 
(
    .clk(clk),
    .rst(rst),

    // AXI input
    .input_axis_tdata(s_axis_tdata),
    .input_axis_tvalid(s_axis_tvalid),
    .input_axis_tready(s_axis_tready),//output, we can assume the signal is always 1 in  normal condition, e.g the fifo is not full
    // AXI output
    .output_axis_tdata(raw_data),
    .output_axis_tvalid(valid_temp),
    .output_axis_tready(raw_ready)

);
assign raw_valid = raw_ready && valid_temp;

endmodule
