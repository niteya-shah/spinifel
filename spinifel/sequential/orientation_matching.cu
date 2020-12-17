#include <cfloat>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
using namespace std::chrono;
namespace py = pybind11;

typedef float dtype;
const int TILE = 16;
const int imgPerThread = 6;

//CUDA kernel used to computed Euclidean distance across data images and reference images.
__global__ void computeEuDistSharReg(float *dataImgs, float *refImgs, float *euDist, int numDataImgs, int numRefImgs, long int totalPixels)
{
        int txId = threadIdx.x;
        int tyId = threadIdx.y;
        int bxId = blockIdx.x;
        int byId = blockIdx.y;

        //Shared variables
        __shared__ dtype sImg0[TILE][TILE+1];
        __shared__ dtype sMod0[TILE][TILE+1];
        __shared__ dtype sImg1[TILE][TILE+1];
        __shared__ dtype sMod1[TILE][TILE+1];
        __shared__ dtype sImg2[TILE][TILE+1];
        __shared__ dtype sMod2[TILE][TILE+1];
        __shared__ dtype sImg3[TILE][TILE+1];
        __shared__ dtype sMod3[TILE][TILE+1];
        __shared__ dtype sImg4[TILE][TILE+1];
        __shared__ dtype sMod4[TILE][TILE+1];
        __shared__ dtype sImg5[TILE][TILE+1];
        __shared__ dtype sMod5[TILE][TILE+1];

        //Register variables
        dtype rImg_0_0,rImg_0_1,rImg_0_2,rImg_0_3,rImg_0_4,rImg_0_5,rImg_0_6,rImg_0_7;
        dtype rImg_0_8,rImg_0_9,rImg_0_10,rImg_0_11,rImg_0_12,rImg_0_13,rImg_0_14,rImg_0_15;
        dtype rMod_0_0,rMod_0_1,rMod_0_2,rMod_0_3,rMod_0_4,rMod_0_5,rMod_0_6,rMod_0_7;
        dtype rMod_0_8,rMod_0_9,rMod_0_10,rMod_0_11,rMod_0_12,rMod_0_13,rMod_0_14,rMod_0_15;
        dtype rImg_1_0,rImg_1_1,rImg_1_2,rImg_1_3,rImg_1_4,rImg_1_5,rImg_1_6,rImg_1_7;
        dtype rImg_1_8,rImg_1_9,rImg_1_10,rImg_1_11,rImg_1_12,rImg_1_13,rImg_1_14,rImg_1_15;
        dtype rMod_1_0,rMod_1_1,rMod_1_2,rMod_1_3,rMod_1_4,rMod_1_5,rMod_1_6,rMod_1_7;
        dtype rMod_1_8,rMod_1_9,rMod_1_10,rMod_1_11,rMod_1_12,rMod_1_13,rMod_1_14,rMod_1_15;
        dtype rImg_2_0,rImg_2_1,rImg_2_2,rImg_2_3,rImg_2_4,rImg_2_5,rImg_2_6,rImg_2_7;
        dtype rImg_2_8,rImg_2_9,rImg_2_10,rImg_2_11,rImg_2_12,rImg_2_13,rImg_2_14,rImg_2_15;
        dtype rMod_2_0,rMod_2_1,rMod_2_2,rMod_2_3,rMod_2_4,rMod_2_5,rMod_2_6,rMod_2_7;
        dtype rMod_2_8,rMod_2_9,rMod_2_10,rMod_2_11,rMod_2_12,rMod_2_13,rMod_2_14,rMod_2_15;
        dtype rImg_3_0,rImg_3_1,rImg_3_2,rImg_3_3,rImg_3_4,rImg_3_5,rImg_3_6,rImg_3_7;
        dtype rImg_3_8,rImg_3_9,rImg_3_10,rImg_3_11,rImg_3_12,rImg_3_13,rImg_3_14,rImg_3_15;
        dtype rMod_3_0,rMod_3_1,rMod_3_2,rMod_3_3,rMod_3_4,rMod_3_5,rMod_3_6,rMod_3_7;
        dtype rMod_3_8,rMod_3_9,rMod_3_10,rMod_3_11,rMod_3_12,rMod_3_13,rMod_3_14,rMod_3_15;
        dtype rImg_4_0,rImg_4_1,rImg_4_2,rImg_4_3,rImg_4_4,rImg_4_5,rImg_4_6,rImg_4_7;
        dtype rImg_4_8,rImg_4_9,rImg_4_10,rImg_4_11,rImg_4_12,rImg_4_13,rImg_4_14,rImg_4_15;
        dtype rMod_4_0,rMod_4_1,rMod_4_2,rMod_4_3,rMod_4_4,rMod_4_5,rMod_4_6,rMod_4_7;
        dtype rMod_4_8,rMod_4_9,rMod_4_10,rMod_4_11,rMod_4_12,rMod_4_13,rMod_4_14,rMod_4_15;
        dtype rImg_5_0,rImg_5_1,rImg_5_2,rImg_5_3,rImg_5_4,rImg_5_5,rImg_5_6,rImg_5_7;
        dtype rImg_5_8,rImg_5_9,rImg_5_10,rImg_5_11,rImg_5_12,rImg_5_13,rImg_5_14,rImg_5_15;
        dtype rMod_5_0,rMod_5_1,rMod_5_2,rMod_5_3,rMod_5_4,rMod_5_5,rMod_5_6,rMod_5_7;
        dtype rMod_5_8,rMod_5_9,rMod_5_10,rMod_5_11,rMod_5_12,rMod_5_13,rMod_5_14,rMod_5_15;

	//euclidean distance variable in each thread.
        float dist00 = 0.0, dist01 = 0.0, dist02 = 0.0, dist03 = 0.0, dist04 = 0.0, dist05 = 0.0;
        float dist10 = 0.0, dist11 = 0.0, dist12 = 0.0, dist13 = 0.0, dist14 = 0.0, dist15 = 0.0;
        float dist20 = 0.0, dist21 = 0.0, dist22 = 0.0, dist23 = 0.0, dist24 = 0.0, dist25 = 0.0;
        float dist30 = 0.0, dist31 = 0.0, dist32 = 0.0, dist33 = 0.0, dist34 = 0.0, dist35 = 0.0;
        float dist40 = 0.0, dist41 = 0.0, dist42 = 0.0, dist43 = 0.0, dist44 = 0.0, dist45 = 0.0;
        float dist50 = 0.0, dist51 = 0.0, dist52 = 0.0, dist53 = 0.0, dist54 = 0.0, dist55 = 0.0;
        int y0Id,yRefs0Id,y1Id,yRefs1Id,y2Id,yRefs2Id,y3Id,yRefs3Id,y4Id,yRefs4Id,y5Id,yRefs5Id;

        int valx = (bxId*TILE*imgPerThread); int valy = (byId*TILE*imgPerThread);
        //1st Row
        if((valy+(0*TILE)+tyId)>=numDataImgs) y0Id= (valy+(0*TILE)+0);
        if((valy+(0*TILE)+tyId)<numDataImgs)  y0Id= (valy+(0*TILE)+tyId);
        if((valx+(0*TILE)+tyId)>=numRefImgs) yRefs0Id= (valx+(0*TILE)+0);
        if((valx+(0*TILE)+tyId)<numRefImgs)  yRefs0Id= (valx+(0*TILE)+tyId);
        //2nd Row
        if((valy+(1*TILE)+tyId)>=numDataImgs) y1Id= (valy+(0*TILE)+0);
        if((valy+(1*TILE)+tyId)<numDataImgs)  y1Id= (valy+(1*TILE)+tyId);
        if((valx+(1*TILE)+tyId)>=numRefImgs) yRefs1Id= (valx+(0*TILE)+0);
        if((valx+(1*TILE)+tyId)<numRefImgs)  yRefs1Id= (valx+(1*TILE)+tyId);
        //3rd Row
        if((valy+(2*TILE)+tyId)>=numDataImgs) y2Id= (valy+(0*TILE)+0);
        if((valy+(2*TILE)+tyId)<numDataImgs)  y2Id= (valy+(2*TILE)+tyId);
        if((valx+(2*TILE)+tyId)>=numRefImgs) yRefs2Id= (valx+(0*TILE)+0);
        if((valx+(2*TILE)+tyId)<numRefImgs)  yRefs2Id= (valx+(2*TILE)+tyId);
        //4th Row
        if((valy+(3*TILE)+tyId)>=numDataImgs) y3Id= (valy+(0*TILE)+0);
        if((valy+(3*TILE)+tyId)<numDataImgs)  y3Id= (valy+(3*TILE)+tyId);
        if((valx+(3*TILE)+tyId)>=numRefImgs) yRefs3Id= (valx+(0*TILE)+0);
        if((valx+(3*TILE)+tyId)<numRefImgs)  yRefs3Id= (valx+(3*TILE)+tyId);
        //5th Row
        if((valy+(4*TILE)+tyId)>=numDataImgs) y4Id= (valy+(0*TILE)+0);
        if((valy+(4*TILE)+tyId)<numDataImgs)  y4Id= (valy+(4*TILE)+tyId);
        if((valx+(4*TILE)+tyId)>=numRefImgs) yRefs4Id= (valx+(0*TILE)+0);
        if((valx+(4*TILE)+tyId)<numRefImgs)  yRefs4Id= (valx+(4*TILE)+tyId);
        //6th Row
        if((valy+(5*TILE)+tyId)>=numDataImgs) y5Id= (valy+(0*TILE)+0);
        if((valy+(5*TILE)+tyId)<numDataImgs)  y5Id= (valy+(5*TILE)+tyId);
        if((valx+(5*TILE)+tyId)>=numRefImgs) yRefs5Id= (valx+(0*TILE)+0);
        if((valx+(5*TILE)+tyId)<numRefImgs)  yRefs5Id= (valx+(5*TILE)+tyId);

        for (int tileIter = 0; tileIter < (totalPixels/TILE); tileIter++)
        {
                sImg0[tyId][txId] = dataImgs[(y0Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod0[tyId][txId] = refImgs[(yRefs0Id*totalPixels) + ((tileIter*TILE) + txId)];
                sImg1[tyId][txId] = dataImgs[(y1Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod1[tyId][txId] = refImgs[(yRefs1Id*totalPixels) + ((tileIter*TILE) + txId)];
                sImg2[tyId][txId] = dataImgs[(y2Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod2[tyId][txId] = refImgs[(yRefs2Id*totalPixels) + ((tileIter*TILE) + txId)];
                sImg3[tyId][txId] = dataImgs[(y3Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod3[tyId][txId] = refImgs[(yRefs3Id*totalPixels) + ((tileIter*TILE) + txId)];
                sImg4[tyId][txId] = dataImgs[(y4Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod4[tyId][txId] = refImgs[(yRefs4Id*totalPixels) + ((tileIter*TILE) + txId)];
                sImg5[tyId][txId] = dataImgs[(y5Id      *totalPixels) + ((tileIter*TILE) + txId)];
                sMod5[tyId][txId] = refImgs[(yRefs5Id*totalPixels) + ((tileIter*TILE) + txId)];

                __syncthreads();
                //1st Row
                rImg_0_0=sImg0[tyId][0];rImg_0_1=sImg0[tyId][1];rImg_0_2=sImg0[tyId][2];rImg_0_3=sImg0[tyId][3];
                rImg_0_4=sImg0[tyId][4];rImg_0_5=sImg0[tyId][5];rImg_0_6=sImg0[tyId][6];rImg_0_7=sImg0[tyId][7];
                rImg_0_8=sImg0[tyId][8];rImg_0_9=sImg0[tyId][9];rImg_0_10=sImg0[tyId][10];rImg_0_11=sImg0[tyId][11];
                rImg_0_12=sImg0[tyId][12];rImg_0_13=sImg0[tyId][13];rImg_0_14=sImg0[tyId][14];rImg_0_15=sImg0[tyId][15];
                rMod_0_0=sMod0[txId][0];rMod_0_1=sMod0[txId][1];rMod_0_2=sMod0[txId][2];rMod_0_3=sMod0[txId][3];
                rMod_0_4=sMod0[txId][4];rMod_0_5=sMod0[txId][5];rMod_0_6=sMod0[txId][6];rMod_0_7=sMod0[txId][7];
                rMod_0_8=sMod0[txId][8];rMod_0_9=sMod0[txId][9];rMod_0_10=sMod0[txId][10];rMod_0_11=sMod0[txId][11];
                rMod_0_12=sMod0[txId][12];rMod_0_13=sMod0[txId][13];rMod_0_14=sMod0[txId][14];rMod_0_15=sMod0[txId][15];
                //2nd Row
                rImg_1_0=sImg1[tyId][0];rImg_1_1=sImg1[tyId][1];rImg_1_2=sImg1[tyId][2];rImg_1_3=sImg1[tyId][3];
                rImg_1_4=sImg1[tyId][4];rImg_1_5=sImg1[tyId][5];rImg_1_6=sImg1[tyId][6];rImg_1_7=sImg1[tyId][7];
                rImg_1_8=sImg1[tyId][8];rImg_1_9=sImg1[tyId][9];rImg_1_10=sImg1[tyId][10];rImg_1_11=sImg1[tyId][11];
                rImg_1_12=sImg1[tyId][12];rImg_1_13=sImg1[tyId][13];rImg_1_14=sImg1[tyId][14];rImg_1_15=sImg1[tyId][15];
                rMod_1_0=sMod1[txId][0];rMod_1_1=sMod1[txId][1];rMod_1_2=sMod1[txId][2];rMod_1_3=sMod1[txId][3];
                rMod_1_4=sMod1[txId][4];rMod_1_5=sMod1[txId][5];rMod_1_6=sMod1[txId][6];rMod_1_7=sMod1[txId][7];
                rMod_1_8=sMod1[txId][8];rMod_1_9=sMod1[txId][9];rMod_1_10=sMod1[txId][10];rMod_1_11=sMod1[txId][11];
                rMod_1_12=sMod1[txId][12];rMod_1_13=sMod1[txId][13];rMod_1_14=sMod1[txId][14];rMod_1_15=sMod1[txId][15];
                //3rd Row
                rImg_2_0=sImg2[tyId][0];rImg_2_1=sImg2[tyId][1];rImg_2_2=sImg2[tyId][2];rImg_2_3=sImg2[tyId][3];
                rImg_2_4=sImg2[tyId][4];rImg_2_5=sImg2[tyId][5];rImg_2_6=sImg2[tyId][6];rImg_2_7=sImg2[tyId][7];
                rImg_2_8=sImg2[tyId][8];rImg_2_9=sImg2[tyId][9];rImg_2_10=sImg2[tyId][10];rImg_2_11=sImg2[tyId][11];
                rImg_2_12=sImg2[tyId][12];rImg_2_13=sImg2[tyId][13];rImg_2_14=sImg2[tyId][14];rImg_2_15=sImg2[tyId][15];
                rMod_2_0=sMod2[txId][0];rMod_2_1=sMod2[txId][1];rMod_2_2=sMod2[txId][2];rMod_2_3=sMod2[txId][3];
                rMod_2_4=sMod2[txId][4];rMod_2_5=sMod2[txId][5];rMod_2_6=sMod2[txId][6];rMod_2_7=sMod2[txId][7];
                rMod_2_8=sMod2[txId][8];rMod_2_9=sMod2[txId][9];rMod_2_10=sMod2[txId][10];rMod_2_11=sMod2[txId][11];
                rMod_2_12=sMod2[txId][12];rMod_2_13=sMod2[txId][13];rMod_2_14=sMod2[txId][14];rMod_2_15=sMod2[txId][15];
                //4th Row
                rImg_3_0=sImg3[tyId][0];rImg_3_1=sImg3[tyId][1];rImg_3_2=sImg3[tyId][2];rImg_3_3=sImg3[tyId][3];
                rImg_3_4=sImg3[tyId][4];rImg_3_5=sImg3[tyId][5];rImg_3_6=sImg3[tyId][6];rImg_3_7=sImg3[tyId][7];
                rImg_3_8=sImg3[tyId][8];rImg_3_9=sImg3[tyId][9];rImg_3_10=sImg3[tyId][10];rImg_3_11=sImg3[tyId][11];
                rImg_3_12=sImg3[tyId][12];rImg_3_13=sImg3[tyId][13];rImg_3_14=sImg3[tyId][14];rImg_3_15=sImg3[tyId][15];
                rMod_3_0=sMod3[txId][0];rMod_3_1=sMod3[txId][1];rMod_3_2=sMod3[txId][2];rMod_3_3=sMod3[txId][3];
                rMod_3_4=sMod3[txId][4];rMod_3_5=sMod3[txId][5];rMod_3_6=sMod3[txId][6];rMod_3_7=sMod3[txId][7];
                rMod_3_8=sMod3[txId][8];rMod_3_9=sMod3[txId][9];rMod_3_10=sMod3[txId][10];rMod_3_11=sMod3[txId][11];
                rMod_3_12=sMod3[txId][12];rMod_3_13=sMod3[txId][13];rMod_3_14=sMod3[txId][14];rMod_3_15=sMod3[txId][15];
                //5th Row
                rImg_4_0=sImg4[tyId][0];rImg_4_1=sImg4[tyId][1];rImg_4_2=sImg4[tyId][2];rImg_4_3=sImg4[tyId][3];
                rImg_4_4=sImg4[tyId][4];rImg_4_5=sImg4[tyId][5];rImg_4_6=sImg4[tyId][6];rImg_4_7=sImg4[tyId][7];
                rImg_4_8=sImg4[tyId][8];rImg_4_9=sImg4[tyId][9];rImg_4_10=sImg4[tyId][10];rImg_4_11=sImg4[tyId][11];
                rImg_4_12=sImg4[tyId][12];rImg_4_13=sImg4[tyId][13];rImg_4_14=sImg4[tyId][14];rImg_4_15=sImg4[tyId][15];
                rMod_4_0=sMod4[txId][0];rMod_4_1=sMod4[txId][1];rMod_4_2=sMod4[txId][2];rMod_4_3=sMod4[txId][3];
                rMod_4_4=sMod4[txId][4];rMod_4_5=sMod4[txId][5];rMod_4_6=sMod4[txId][6];rMod_4_7=sMod4[txId][7];
                rMod_4_8=sMod4[txId][8];rMod_4_9=sMod4[txId][9];rMod_4_10=sMod4[txId][10];rMod_4_11=sMod4[txId][11];
                rMod_4_12=sMod4[txId][12];rMod_4_13=sMod4[txId][13];rMod_4_14=sMod4[txId][14];rMod_4_15=sMod4[txId][15];
                //6th Row
                rImg_5_0=sImg5[tyId][0];rImg_5_1=sImg5[tyId][1];rImg_5_2=sImg5[tyId][2];rImg_5_3=sImg5[tyId][3];
                rImg_5_4=sImg5[tyId][4];rImg_5_5=sImg5[tyId][5];rImg_5_6=sImg5[tyId][6];rImg_5_7=sImg5[tyId][7];
                rImg_5_8=sImg5[tyId][8];rImg_5_9=sImg5[tyId][9];rImg_5_10=sImg5[tyId][10];rImg_5_11=sImg5[tyId][11];
                rImg_5_12=sImg5[tyId][12];rImg_5_13=sImg5[tyId][13];rImg_5_14=sImg5[tyId][14];rImg_5_15=sImg5[tyId][15];
                rMod_5_0=sMod5[txId][0];rMod_5_1=sMod5[txId][1];rMod_5_2=sMod5[txId][2];rMod_5_3=sMod5[txId][3];
                rMod_5_4=sMod5[txId][4];rMod_5_5=sMod5[txId][5];rMod_5_6=sMod5[txId][6];rMod_5_7=sMod5[txId][7];
                rMod_5_8=sMod5[txId][8];rMod_5_9=sMod5[txId][9];rMod_5_10=sMod5[txId][10];rMod_5_11=sMod5[txId][11];
                rMod_5_12=sMod5[txId][12];rMod_5_13=sMod5[txId][13];rMod_5_14=sMod5[txId][14];rMod_5_15=sMod5[txId][15];

                //1st Row
                dist00 = dist00 + ((rImg_0_0-rMod_0_0)*(rImg_0_0-rMod_0_0)) + ((rImg_0_1-rMod_0_1)*(rImg_0_1-rMod_0_1)) +
                                  ((rImg_0_2-rMod_0_2)*(rImg_0_2-rMod_0_2)) + ((rImg_0_3-rMod_0_3)*(rImg_0_3-rMod_0_3)) +
                                  ((rImg_0_4-rMod_0_4)*(rImg_0_4-rMod_0_4)) + ((rImg_0_5-rMod_0_5)*(rImg_0_5-rMod_0_5)) +
                                  ((rImg_0_6-rMod_0_6)*(rImg_0_6-rMod_0_6)) + ((rImg_0_7-rMod_0_7)*(rImg_0_7-rMod_0_7)) +
                                  ((rImg_0_8-rMod_0_8)*(rImg_0_8-rMod_0_8)) + ((rImg_0_9-rMod_0_9)*(rImg_0_9-rMod_0_9)) +
                                  ((rImg_0_10-rMod_0_10)*(rImg_0_10-rMod_0_10)) + ((rImg_0_11-rMod_0_11)*(rImg_0_11-rMod_0_11)) +
                                  ((rImg_0_12-rMod_0_12)*(rImg_0_12-rMod_0_12)) + ((rImg_0_13-rMod_0_13)*(rImg_0_13-rMod_0_13)) +
                                  ((rImg_0_14-rMod_0_14)*(rImg_0_14-rMod_0_14)) + ((rImg_0_15-rMod_0_15)*(rImg_0_15-rMod_0_15));
                dist01 = dist01 + ((rImg_0_0-rMod_1_0)*(rImg_0_0-rMod_1_0)) + ((rImg_0_1-rMod_1_1)*(rImg_0_1-rMod_1_1)) +
                                  ((rImg_0_2-rMod_1_2)*(rImg_0_2-rMod_1_2)) + ((rImg_0_3-rMod_1_3)*(rImg_0_3-rMod_1_3)) +
                                  ((rImg_0_4-rMod_1_4)*(rImg_0_4-rMod_1_4)) + ((rImg_0_5-rMod_1_5)*(rImg_0_5-rMod_1_5)) +
                                  ((rImg_0_6-rMod_1_6)*(rImg_0_6-rMod_1_6)) + ((rImg_0_7-rMod_1_7)*(rImg_0_7-rMod_1_7)) +
                                  ((rImg_0_8-rMod_1_8)*(rImg_0_8-rMod_1_8)) + ((rImg_0_9-rMod_1_9)*(rImg_0_9-rMod_1_9)) +
                                  ((rImg_0_10-rMod_1_10)*(rImg_0_10-rMod_1_10)) + ((rImg_0_11-rMod_1_11)*(rImg_0_11-rMod_1_11)) +
                                  ((rImg_0_12-rMod_1_12)*(rImg_0_12-rMod_1_12)) + ((rImg_0_13-rMod_1_13)*(rImg_0_13-rMod_1_13)) +
                                  ((rImg_0_14-rMod_1_14)*(rImg_0_14-rMod_1_14)) + ((rImg_0_15-rMod_1_15)*(rImg_0_15-rMod_1_15));
                dist02 = dist02 + ((rImg_0_0-rMod_2_0)*(rImg_0_0-rMod_2_0)) + ((rImg_0_1-rMod_2_1)*(rImg_0_1-rMod_2_1)) +
                                  ((rImg_0_2-rMod_2_2)*(rImg_0_2-rMod_2_2)) + ((rImg_0_3-rMod_2_3)*(rImg_0_3-rMod_2_3)) +
                                  ((rImg_0_4-rMod_2_4)*(rImg_0_4-rMod_2_4)) + ((rImg_0_5-rMod_2_5)*(rImg_0_5-rMod_2_5)) +
                                  ((rImg_0_6-rMod_2_6)*(rImg_0_6-rMod_2_6)) + ((rImg_0_7-rMod_2_7)*(rImg_0_7-rMod_2_7)) +
                                  ((rImg_0_8-rMod_2_8)*(rImg_0_8-rMod_2_8)) + ((rImg_0_9-rMod_2_9)*(rImg_0_9-rMod_2_9)) +
                                  ((rImg_0_10-rMod_2_10)*(rImg_0_10-rMod_2_10)) + ((rImg_0_11-rMod_2_11)*(rImg_0_11-rMod_2_11)) +
                                  ((rImg_0_12-rMod_2_12)*(rImg_0_12-rMod_2_12)) + ((rImg_0_13-rMod_2_13)*(rImg_0_13-rMod_2_13)) +
                                  ((rImg_0_14-rMod_2_14)*(rImg_0_14-rMod_2_14)) + ((rImg_0_15-rMod_2_15)*(rImg_0_15-rMod_2_15));
                dist03 = dist03 + ((rImg_0_0-rMod_3_0)*(rImg_0_0-rMod_3_0)) + ((rImg_0_1-rMod_3_1)*(rImg_0_1-rMod_3_1)) +
                                  ((rImg_0_2-rMod_3_2)*(rImg_0_2-rMod_3_2)) + ((rImg_0_3-rMod_3_3)*(rImg_0_3-rMod_3_3)) +
                                  ((rImg_0_4-rMod_3_4)*(rImg_0_4-rMod_3_4)) + ((rImg_0_5-rMod_3_5)*(rImg_0_5-rMod_3_5)) +
                                  ((rImg_0_6-rMod_3_6)*(rImg_0_6-rMod_3_6)) + ((rImg_0_7-rMod_3_7)*(rImg_0_7-rMod_3_7)) +
                                  ((rImg_0_8-rMod_3_8)*(rImg_0_8-rMod_3_8)) + ((rImg_0_9-rMod_3_9)*(rImg_0_9-rMod_3_9)) +
                                  ((rImg_0_10-rMod_3_10)*(rImg_0_10-rMod_3_10)) + ((rImg_0_11-rMod_3_11)*(rImg_0_11-rMod_3_11)) +
                                  ((rImg_0_12-rMod_3_12)*(rImg_0_12-rMod_3_12)) + ((rImg_0_13-rMod_3_13)*(rImg_0_13-rMod_3_13)) +
                                  ((rImg_0_14-rMod_3_14)*(rImg_0_14-rMod_3_14)) + ((rImg_0_15-rMod_3_15)*(rImg_0_15-rMod_3_15));
                dist04 = dist04 + ((rImg_0_0-rMod_4_0)*(rImg_0_0-rMod_4_0)) + ((rImg_0_1-rMod_4_1)*(rImg_0_1-rMod_4_1)) +
                                  ((rImg_0_2-rMod_4_2)*(rImg_0_2-rMod_4_2)) + ((rImg_0_3-rMod_4_3)*(rImg_0_3-rMod_4_3)) +
                                  ((rImg_0_4-rMod_4_4)*(rImg_0_4-rMod_4_4)) + ((rImg_0_5-rMod_4_5)*(rImg_0_5-rMod_4_5)) +
                                  ((rImg_0_6-rMod_4_6)*(rImg_0_6-rMod_4_6)) + ((rImg_0_7-rMod_4_7)*(rImg_0_7-rMod_4_7)) +
                                  ((rImg_0_8-rMod_4_8)*(rImg_0_8-rMod_4_8)) + ((rImg_0_9-rMod_4_9)*(rImg_0_9-rMod_4_9)) +
                                  ((rImg_0_10-rMod_4_10)*(rImg_0_10-rMod_4_10)) + ((rImg_0_11-rMod_4_11)*(rImg_0_11-rMod_4_11)) +
                                  ((rImg_0_12-rMod_4_12)*(rImg_0_12-rMod_4_12)) + ((rImg_0_13-rMod_4_13)*(rImg_0_13-rMod_4_13)) +
                                  ((rImg_0_14-rMod_4_14)*(rImg_0_14-rMod_4_14)) + ((rImg_0_15-rMod_4_15)*(rImg_0_15-rMod_4_15));
                dist05 = dist05 + ((rImg_0_0-rMod_5_0)*(rImg_0_0-rMod_5_0)) + ((rImg_0_1-rMod_5_1)*(rImg_0_1-rMod_5_1)) +
                                  ((rImg_0_2-rMod_5_2)*(rImg_0_2-rMod_5_2)) + ((rImg_0_3-rMod_5_3)*(rImg_0_3-rMod_5_3)) +
                                  ((rImg_0_4-rMod_5_4)*(rImg_0_4-rMod_5_4)) + ((rImg_0_5-rMod_5_5)*(rImg_0_5-rMod_5_5)) +
                                  ((rImg_0_6-rMod_5_6)*(rImg_0_6-rMod_5_6)) + ((rImg_0_7-rMod_5_7)*(rImg_0_7-rMod_5_7)) +
                                  ((rImg_0_8-rMod_5_8)*(rImg_0_8-rMod_5_8)) + ((rImg_0_9-rMod_5_9)*(rImg_0_9-rMod_5_9)) +
                                  ((rImg_0_10-rMod_5_10)*(rImg_0_10-rMod_5_10)) + ((rImg_0_11-rMod_5_11)*(rImg_0_11-rMod_5_11)) +
                                  ((rImg_0_12-rMod_5_12)*(rImg_0_12-rMod_5_12)) + ((rImg_0_13-rMod_5_13)*(rImg_0_13-rMod_5_13)) +
                                  ((rImg_0_14-rMod_5_14)*(rImg_0_14-rMod_5_14)) + ((rImg_0_15-rMod_5_15)*(rImg_0_15-rMod_5_15));
                //2nd Row
                dist10 = dist10 + ((rImg_1_0-rMod_0_0)*(rImg_1_0-rMod_0_0)) + ((rImg_1_1-rMod_0_1)*(rImg_1_1-rMod_0_1)) +
                                  ((rImg_1_2-rMod_0_2)*(rImg_1_2-rMod_0_2)) + ((rImg_1_3-rMod_0_3)*(rImg_1_3-rMod_0_3)) +
                                  ((rImg_1_4-rMod_0_4)*(rImg_1_4-rMod_0_4)) + ((rImg_1_5-rMod_0_5)*(rImg_1_5-rMod_0_5)) +
                                  ((rImg_1_6-rMod_0_6)*(rImg_1_6-rMod_0_6)) + ((rImg_1_7-rMod_0_7)*(rImg_1_7-rMod_0_7)) +
                                  ((rImg_1_8-rMod_0_8)*(rImg_1_8-rMod_0_8)) + ((rImg_1_9-rMod_0_9)*(rImg_1_9-rMod_0_9)) +
                                  ((rImg_1_10-rMod_0_10)*(rImg_1_10-rMod_0_10)) + ((rImg_1_11-rMod_0_11)*(rImg_1_11-rMod_0_11)) +
                                  ((rImg_1_12-rMod_0_12)*(rImg_1_12-rMod_0_12)) + ((rImg_1_13-rMod_0_13)*(rImg_1_13-rMod_0_13)) +
                                  ((rImg_1_14-rMod_0_14)*(rImg_1_14-rMod_0_14)) + ((rImg_1_15-rMod_0_15)*(rImg_1_15-rMod_0_15));
                dist11 = dist11 + ((rImg_1_0-rMod_1_0)*(rImg_1_0-rMod_1_0)) + ((rImg_1_1-rMod_1_1)*(rImg_1_1-rMod_1_1)) +
                                  ((rImg_1_2-rMod_1_2)*(rImg_1_2-rMod_1_2)) + ((rImg_1_3-rMod_1_3)*(rImg_1_3-rMod_1_3)) +
                                  ((rImg_1_4-rMod_1_4)*(rImg_1_4-rMod_1_4)) + ((rImg_1_5-rMod_1_5)*(rImg_1_5-rMod_1_5)) +
                                  ((rImg_1_6-rMod_1_6)*(rImg_1_6-rMod_1_6)) + ((rImg_1_7-rMod_1_7)*(rImg_1_7-rMod_1_7)) +
                                  ((rImg_1_8-rMod_1_8)*(rImg_1_8-rMod_1_8)) + ((rImg_1_9-rMod_1_9)*(rImg_1_9-rMod_1_9)) +
                                  ((rImg_1_10-rMod_1_10)*(rImg_1_10-rMod_1_10)) + ((rImg_1_11-rMod_1_11)*(rImg_1_11-rMod_1_11)) +
                                  ((rImg_1_12-rMod_1_12)*(rImg_1_12-rMod_1_12)) + ((rImg_1_13-rMod_1_13)*(rImg_1_13-rMod_1_13)) +
                                  ((rImg_1_14-rMod_1_14)*(rImg_1_14-rMod_1_14)) + ((rImg_1_15-rMod_1_15)*(rImg_1_15-rMod_1_15));
                dist12 = dist12 + ((rImg_1_0-rMod_2_0)*(rImg_1_0-rMod_2_0)) + ((rImg_1_1-rMod_2_1)*(rImg_1_1-rMod_2_1)) +
                                  ((rImg_1_2-rMod_2_2)*(rImg_1_2-rMod_2_2)) + ((rImg_1_3-rMod_2_3)*(rImg_1_3-rMod_2_3)) +
                                  ((rImg_1_4-rMod_2_4)*(rImg_1_4-rMod_2_4)) + ((rImg_1_5-rMod_2_5)*(rImg_1_5-rMod_2_5)) +
                                  ((rImg_1_6-rMod_2_6)*(rImg_1_6-rMod_2_6)) + ((rImg_1_7-rMod_2_7)*(rImg_1_7-rMod_2_7)) +
                                  ((rImg_1_8-rMod_2_8)*(rImg_1_8-rMod_2_8)) + ((rImg_1_9-rMod_2_9)*(rImg_1_9-rMod_2_9)) +
                                  ((rImg_1_10-rMod_2_10)*(rImg_1_10-rMod_2_10)) + ((rImg_1_11-rMod_2_11)*(rImg_1_11-rMod_2_11)) +
                                  ((rImg_1_12-rMod_2_12)*(rImg_1_12-rMod_2_12)) + ((rImg_1_13-rMod_2_13)*(rImg_1_13-rMod_2_13)) +
                                  ((rImg_1_14-rMod_2_14)*(rImg_1_14-rMod_2_14)) + ((rImg_1_15-rMod_2_15)*(rImg_1_15-rMod_2_15));
                dist13 = dist13 + ((rImg_1_0-rMod_3_0)*(rImg_1_0-rMod_3_0)) + ((rImg_1_1-rMod_3_1)*(rImg_1_1-rMod_3_1)) +
                                  ((rImg_1_2-rMod_3_2)*(rImg_1_2-rMod_3_2)) + ((rImg_1_3-rMod_3_3)*(rImg_1_3-rMod_3_3)) +
                                  ((rImg_1_4-rMod_3_4)*(rImg_1_4-rMod_3_4)) + ((rImg_1_5-rMod_3_5)*(rImg_1_5-rMod_3_5)) +
                                  ((rImg_1_6-rMod_3_6)*(rImg_1_6-rMod_3_6)) + ((rImg_1_7-rMod_3_7)*(rImg_1_7-rMod_3_7)) +
                                  ((rImg_1_8-rMod_3_8)*(rImg_1_8-rMod_3_8)) + ((rImg_1_9-rMod_3_9)*(rImg_1_9-rMod_3_9)) +
                                  ((rImg_1_10-rMod_3_10)*(rImg_1_10-rMod_3_10)) + ((rImg_1_11-rMod_3_11)*(rImg_1_11-rMod_3_11)) +
                                  ((rImg_1_12-rMod_3_12)*(rImg_1_12-rMod_3_12)) + ((rImg_1_13-rMod_3_13)*(rImg_1_13-rMod_3_13)) +
                                  ((rImg_1_14-rMod_3_14)*(rImg_1_14-rMod_3_14)) + ((rImg_1_15-rMod_3_15)*(rImg_1_15-rMod_3_15));
                dist14 = dist14 + ((rImg_1_0-rMod_4_0)*(rImg_1_0-rMod_4_0)) + ((rImg_1_1-rMod_4_1)*(rImg_1_1-rMod_4_1)) +
                                  ((rImg_1_2-rMod_4_2)*(rImg_1_2-rMod_4_2)) + ((rImg_1_3-rMod_4_3)*(rImg_1_3-rMod_4_3)) +
                                  ((rImg_1_4-rMod_4_4)*(rImg_1_4-rMod_4_4)) + ((rImg_1_5-rMod_4_5)*(rImg_1_5-rMod_4_5)) +
                                  ((rImg_1_6-rMod_4_6)*(rImg_1_6-rMod_4_6)) + ((rImg_1_7-rMod_4_7)*(rImg_1_7-rMod_4_7)) +
                                  ((rImg_1_8-rMod_4_8)*(rImg_1_8-rMod_4_8)) + ((rImg_1_9-rMod_4_9)*(rImg_1_9-rMod_4_9)) +
                                  ((rImg_1_10-rMod_4_10)*(rImg_1_10-rMod_4_10)) + ((rImg_1_11-rMod_4_11)*(rImg_1_11-rMod_4_11)) +
                                  ((rImg_1_12-rMod_4_12)*(rImg_1_12-rMod_4_12)) + ((rImg_1_13-rMod_4_13)*(rImg_1_13-rMod_4_13)) +
                                  ((rImg_1_14-rMod_4_14)*(rImg_1_14-rMod_4_14)) + ((rImg_1_15-rMod_4_15)*(rImg_1_15-rMod_4_15));
                dist15 = dist15 + ((rImg_1_0-rMod_5_0)*(rImg_1_0-rMod_5_0)) + ((rImg_1_1-rMod_5_1)*(rImg_1_1-rMod_5_1)) +
                                  ((rImg_1_2-rMod_5_2)*(rImg_1_2-rMod_5_2)) + ((rImg_1_3-rMod_5_3)*(rImg_1_3-rMod_5_3)) +
                                  ((rImg_1_4-rMod_5_4)*(rImg_1_4-rMod_5_4)) + ((rImg_1_5-rMod_5_5)*(rImg_1_5-rMod_5_5)) +
                                  ((rImg_1_6-rMod_5_6)*(rImg_1_6-rMod_5_6)) + ((rImg_1_7-rMod_5_7)*(rImg_1_7-rMod_5_7)) +
                                  ((rImg_1_8-rMod_5_8)*(rImg_1_8-rMod_5_8)) + ((rImg_1_9-rMod_5_9)*(rImg_1_9-rMod_5_9)) +
                                  ((rImg_1_10-rMod_5_10)*(rImg_1_10-rMod_5_10)) + ((rImg_1_11-rMod_5_11)*(rImg_1_11-rMod_5_11)) +
                                  ((rImg_1_12-rMod_5_12)*(rImg_1_12-rMod_5_12)) + ((rImg_1_13-rMod_5_13)*(rImg_1_13-rMod_5_13)) +
                                  ((rImg_1_14-rMod_5_14)*(rImg_1_14-rMod_5_14)) + ((rImg_1_15-rMod_5_15)*(rImg_1_15-rMod_5_15));
                //3rd Row
                dist20 = dist20 + ((rImg_2_0-rMod_0_0)*(rImg_2_0-rMod_0_0)) + ((rImg_2_1-rMod_0_1)*(rImg_2_1-rMod_0_1)) +
                                  ((rImg_2_2-rMod_0_2)*(rImg_2_2-rMod_0_2)) + ((rImg_2_3-rMod_0_3)*(rImg_2_3-rMod_0_3)) +
                                  ((rImg_2_4-rMod_0_4)*(rImg_2_4-rMod_0_4)) + ((rImg_2_5-rMod_0_5)*(rImg_2_5-rMod_0_5)) +
                                  ((rImg_2_6-rMod_0_6)*(rImg_2_6-rMod_0_6)) + ((rImg_2_7-rMod_0_7)*(rImg_2_7-rMod_0_7)) +
                                  ((rImg_2_8-rMod_0_8)*(rImg_2_8-rMod_0_8)) + ((rImg_2_9-rMod_0_9)*(rImg_2_9-rMod_0_9)) +
                                  ((rImg_2_10-rMod_0_10)*(rImg_2_10-rMod_0_10)) + ((rImg_2_11-rMod_0_11)*(rImg_2_11-rMod_0_11)) +
                                  ((rImg_2_12-rMod_0_12)*(rImg_2_12-rMod_0_12)) + ((rImg_2_13-rMod_0_13)*(rImg_2_13-rMod_0_13)) +
                                  ((rImg_2_14-rMod_0_14)*(rImg_2_14-rMod_0_14)) + ((rImg_2_15-rMod_0_15)*(rImg_2_15-rMod_0_15));
                dist21 = dist21 + ((rImg_2_0-rMod_1_0)*(rImg_2_0-rMod_1_0)) + ((rImg_2_1-rMod_1_1)*(rImg_2_1-rMod_1_1)) +
                                  ((rImg_2_2-rMod_1_2)*(rImg_2_2-rMod_1_2)) + ((rImg_2_3-rMod_1_3)*(rImg_2_3-rMod_1_3)) +
                                  ((rImg_2_4-rMod_1_4)*(rImg_2_4-rMod_1_4)) + ((rImg_2_5-rMod_1_5)*(rImg_2_5-rMod_1_5)) +
                                  ((rImg_2_6-rMod_1_6)*(rImg_2_6-rMod_1_6)) + ((rImg_2_7-rMod_1_7)*(rImg_2_7-rMod_1_7)) +
                                  ((rImg_2_8-rMod_1_8)*(rImg_2_8-rMod_1_8)) + ((rImg_2_9-rMod_1_9)*(rImg_2_9-rMod_1_9)) +
                                  ((rImg_2_10-rMod_1_10)*(rImg_2_10-rMod_1_10)) + ((rImg_2_11-rMod_1_11)*(rImg_2_11-rMod_1_11)) +
                                  ((rImg_2_12-rMod_1_12)*(rImg_2_12-rMod_1_12)) + ((rImg_2_13-rMod_1_13)*(rImg_2_13-rMod_1_13)) +
                                  ((rImg_2_14-rMod_1_14)*(rImg_2_14-rMod_1_14)) + ((rImg_2_15-rMod_1_15)*(rImg_2_15-rMod_1_15));
                dist22 = dist22 + ((rImg_2_0-rMod_2_0)*(rImg_2_0-rMod_2_0)) + ((rImg_2_1-rMod_2_1)*(rImg_2_1-rMod_2_1)) +
                                  ((rImg_2_2-rMod_2_2)*(rImg_2_2-rMod_2_2)) + ((rImg_2_3-rMod_2_3)*(rImg_2_3-rMod_2_3)) +
                                  ((rImg_2_4-rMod_2_4)*(rImg_2_4-rMod_2_4)) + ((rImg_2_5-rMod_2_5)*(rImg_2_5-rMod_2_5)) +
                                  ((rImg_2_6-rMod_2_6)*(rImg_2_6-rMod_2_6)) + ((rImg_2_7-rMod_2_7)*(rImg_2_7-rMod_2_7)) +
                                  ((rImg_2_8-rMod_2_8)*(rImg_2_8-rMod_2_8)) + ((rImg_2_9-rMod_2_9)*(rImg_2_9-rMod_2_9)) +
                                  ((rImg_2_10-rMod_2_10)*(rImg_2_10-rMod_2_10)) + ((rImg_2_11-rMod_2_11)*(rImg_2_11-rMod_2_11)) +
                                  ((rImg_2_12-rMod_2_12)*(rImg_2_12-rMod_2_12)) + ((rImg_2_13-rMod_2_13)*(rImg_2_13-rMod_2_13)) +
                                  ((rImg_2_14-rMod_2_14)*(rImg_2_14-rMod_2_14)) + ((rImg_2_15-rMod_2_15)*(rImg_2_15-rMod_2_15));
                dist23 = dist23 + ((rImg_2_0-rMod_3_0)*(rImg_2_0-rMod_3_0)) + ((rImg_2_1-rMod_3_1)*(rImg_2_1-rMod_3_1)) +
                                  ((rImg_2_2-rMod_3_2)*(rImg_2_2-rMod_3_2)) + ((rImg_2_3-rMod_3_3)*(rImg_2_3-rMod_3_3)) +
                                  ((rImg_2_4-rMod_3_4)*(rImg_2_4-rMod_3_4)) + ((rImg_2_5-rMod_3_5)*(rImg_2_5-rMod_3_5)) +
                                  ((rImg_2_6-rMod_3_6)*(rImg_2_6-rMod_3_6)) + ((rImg_2_7-rMod_3_7)*(rImg_2_7-rMod_3_7)) +
                                  ((rImg_2_8-rMod_3_8)*(rImg_2_8-rMod_3_8)) + ((rImg_2_9-rMod_3_9)*(rImg_2_9-rMod_3_9)) +
                                  ((rImg_2_10-rMod_3_10)*(rImg_2_10-rMod_3_10)) + ((rImg_2_11-rMod_3_11)*(rImg_2_11-rMod_3_11)) +
                                  ((rImg_2_12-rMod_3_12)*(rImg_2_12-rMod_3_12)) + ((rImg_2_13-rMod_3_13)*(rImg_2_13-rMod_3_13)) +
                                  ((rImg_2_14-rMod_3_14)*(rImg_2_14-rMod_3_14)) + ((rImg_2_15-rMod_3_15)*(rImg_2_15-rMod_3_15));
                dist24 = dist24 + ((rImg_2_0-rMod_4_0)*(rImg_2_0-rMod_4_0)) + ((rImg_2_1-rMod_4_1)*(rImg_2_1-rMod_4_1)) +
                                  ((rImg_2_2-rMod_4_2)*(rImg_2_2-rMod_4_2)) + ((rImg_2_3-rMod_4_3)*(rImg_2_3-rMod_4_3)) +
                                  ((rImg_2_4-rMod_4_4)*(rImg_2_4-rMod_4_4)) + ((rImg_2_5-rMod_4_5)*(rImg_2_5-rMod_4_5)) +
                                  ((rImg_2_6-rMod_4_6)*(rImg_2_6-rMod_4_6)) + ((rImg_2_7-rMod_4_7)*(rImg_2_7-rMod_4_7)) +
                                  ((rImg_2_8-rMod_4_8)*(rImg_2_8-rMod_4_8)) + ((rImg_2_9-rMod_4_9)*(rImg_2_9-rMod_4_9)) +
                                  ((rImg_2_10-rMod_4_10)*(rImg_2_10-rMod_4_10)) + ((rImg_2_11-rMod_4_11)*(rImg_2_11-rMod_4_11)) +
                                  ((rImg_2_12-rMod_4_12)*(rImg_2_12-rMod_4_12)) + ((rImg_2_13-rMod_4_13)*(rImg_2_13-rMod_4_13)) +
                                  ((rImg_2_14-rMod_4_14)*(rImg_2_14-rMod_4_14)) + ((rImg_2_15-rMod_4_15)*(rImg_2_15-rMod_4_15));
                dist25 = dist25 + ((rImg_2_0-rMod_5_0)*(rImg_2_0-rMod_5_0)) + ((rImg_2_1-rMod_5_1)*(rImg_2_1-rMod_5_1)) +
                                  ((rImg_2_2-rMod_5_2)*(rImg_2_2-rMod_5_2)) + ((rImg_2_3-rMod_5_3)*(rImg_2_3-rMod_5_3)) +
                                  ((rImg_2_4-rMod_5_4)*(rImg_2_4-rMod_5_4)) + ((rImg_2_5-rMod_5_5)*(rImg_2_5-rMod_5_5)) +
                                  ((rImg_2_6-rMod_5_6)*(rImg_2_6-rMod_5_6)) + ((rImg_2_7-rMod_5_7)*(rImg_2_7-rMod_5_7)) +
                                  ((rImg_2_8-rMod_5_8)*(rImg_2_8-rMod_5_8)) + ((rImg_2_9-rMod_5_9)*(rImg_2_9-rMod_5_9)) +
                                  ((rImg_2_10-rMod_5_10)*(rImg_2_10-rMod_5_10)) + ((rImg_2_11-rMod_5_11)*(rImg_2_11-rMod_5_11)) +
                                  ((rImg_2_12-rMod_5_12)*(rImg_2_12-rMod_5_12)) + ((rImg_2_13-rMod_5_13)*(rImg_2_13-rMod_5_13)) +
                                  ((rImg_2_14-rMod_5_14)*(rImg_2_14-rMod_5_14)) + ((rImg_2_15-rMod_5_15)*(rImg_2_15-rMod_5_15));
                //4th Row
                dist30 = dist30 + ((rImg_3_0-rMod_0_0)*(rImg_3_0-rMod_0_0)) + ((rImg_3_1-rMod_0_1)*(rImg_3_1-rMod_0_1)) +
                                  ((rImg_3_2-rMod_0_2)*(rImg_3_2-rMod_0_2)) + ((rImg_3_3-rMod_0_3)*(rImg_3_3-rMod_0_3)) +
                                  ((rImg_3_4-rMod_0_4)*(rImg_3_4-rMod_0_4)) + ((rImg_3_5-rMod_0_5)*(rImg_3_5-rMod_0_5)) +
                                  ((rImg_3_6-rMod_0_6)*(rImg_3_6-rMod_0_6)) + ((rImg_3_7-rMod_0_7)*(rImg_3_7-rMod_0_7)) +
                                  ((rImg_3_8-rMod_0_8)*(rImg_3_8-rMod_0_8)) + ((rImg_3_9-rMod_0_9)*(rImg_3_9-rMod_0_9)) +
                                  ((rImg_3_10-rMod_0_10)*(rImg_3_10-rMod_0_10)) + ((rImg_3_11-rMod_0_11)*(rImg_3_11-rMod_0_11)) +
                                  ((rImg_3_12-rMod_0_12)*(rImg_3_12-rMod_0_12)) + ((rImg_3_13-rMod_0_13)*(rImg_3_13-rMod_0_13)) +
                                  ((rImg_3_14-rMod_0_14)*(rImg_3_14-rMod_0_14)) + ((rImg_3_15-rMod_0_15)*(rImg_3_15-rMod_0_15));
                dist31 = dist31 + ((rImg_3_0-rMod_1_0)*(rImg_3_0-rMod_1_0)) + ((rImg_3_1-rMod_1_1)*(rImg_3_1-rMod_1_1)) +
                                  ((rImg_3_2-rMod_1_2)*(rImg_3_2-rMod_1_2)) + ((rImg_3_3-rMod_1_3)*(rImg_3_3-rMod_1_3)) +
                                  ((rImg_3_4-rMod_1_4)*(rImg_3_4-rMod_1_4)) + ((rImg_3_5-rMod_1_5)*(rImg_3_5-rMod_1_5)) +
                                  ((rImg_3_6-rMod_1_6)*(rImg_3_6-rMod_1_6)) + ((rImg_3_7-rMod_1_7)*(rImg_3_7-rMod_1_7)) +
                                  ((rImg_3_8-rMod_1_8)*(rImg_3_8-rMod_1_8)) + ((rImg_3_9-rMod_1_9)*(rImg_3_9-rMod_1_9)) +
                                  ((rImg_3_10-rMod_1_10)*(rImg_3_10-rMod_1_10)) + ((rImg_3_11-rMod_1_11)*(rImg_3_11-rMod_1_11)) +
                                  ((rImg_3_12-rMod_1_12)*(rImg_3_12-rMod_1_12)) + ((rImg_3_13-rMod_1_13)*(rImg_3_13-rMod_1_13)) +
                                  ((rImg_3_14-rMod_1_14)*(rImg_3_14-rMod_1_14)) + ((rImg_3_15-rMod_1_15)*(rImg_3_15-rMod_1_15));
                dist32 = dist32 + ((rImg_3_0-rMod_2_0)*(rImg_3_0-rMod_2_0)) + ((rImg_3_1-rMod_2_1)*(rImg_3_1-rMod_2_1)) +
                                  ((rImg_3_2-rMod_2_2)*(rImg_3_2-rMod_2_2)) + ((rImg_3_3-rMod_2_3)*(rImg_3_3-rMod_2_3)) +
                                  ((rImg_3_4-rMod_2_4)*(rImg_3_4-rMod_2_4)) + ((rImg_3_5-rMod_2_5)*(rImg_3_5-rMod_2_5)) +
                                  ((rImg_3_6-rMod_2_6)*(rImg_3_6-rMod_2_6)) + ((rImg_3_7-rMod_2_7)*(rImg_3_7-rMod_2_7)) +
                                  ((rImg_3_8-rMod_2_8)*(rImg_3_8-rMod_2_8)) + ((rImg_3_9-rMod_2_9)*(rImg_3_9-rMod_2_9)) +
                                  ((rImg_3_10-rMod_2_10)*(rImg_3_10-rMod_2_10)) + ((rImg_3_11-rMod_2_11)*(rImg_3_11-rMod_2_11)) +
                                  ((rImg_3_12-rMod_2_12)*(rImg_3_12-rMod_2_12)) + ((rImg_3_13-rMod_2_13)*(rImg_3_13-rMod_2_13)) +
                                  ((rImg_3_14-rMod_2_14)*(rImg_3_14-rMod_2_14)) + ((rImg_3_15-rMod_2_15)*(rImg_3_15-rMod_2_15));
                dist33 = dist33 + ((rImg_3_0-rMod_3_0)*(rImg_3_0-rMod_3_0)) + ((rImg_3_1-rMod_3_1)*(rImg_3_1-rMod_3_1)) +
                                  ((rImg_3_2-rMod_3_2)*(rImg_3_2-rMod_3_2)) + ((rImg_3_3-rMod_3_3)*(rImg_3_3-rMod_3_3)) +
                                  ((rImg_3_4-rMod_3_4)*(rImg_3_4-rMod_3_4)) + ((rImg_3_5-rMod_3_5)*(rImg_3_5-rMod_3_5)) +
                                  ((rImg_3_6-rMod_3_6)*(rImg_3_6-rMod_3_6)) + ((rImg_3_7-rMod_3_7)*(rImg_3_7-rMod_3_7)) +
                                  ((rImg_3_8-rMod_3_8)*(rImg_3_8-rMod_3_8)) + ((rImg_3_9-rMod_3_9)*(rImg_3_9-rMod_3_9)) +
                                  ((rImg_3_10-rMod_3_10)*(rImg_3_10-rMod_3_10)) + ((rImg_3_11-rMod_3_11)*(rImg_3_11-rMod_3_11)) +
                                  ((rImg_3_12-rMod_3_12)*(rImg_3_12-rMod_3_12)) + ((rImg_3_13-rMod_3_13)*(rImg_3_13-rMod_3_13)) +
                                  ((rImg_3_14-rMod_3_14)*(rImg_3_14-rMod_3_14)) + ((rImg_3_15-rMod_3_15)*(rImg_3_15-rMod_3_15));
                dist34 = dist34 + ((rImg_3_0-rMod_4_0)*(rImg_3_0-rMod_4_0)) + ((rImg_3_1-rMod_4_1)*(rImg_3_1-rMod_4_1)) +
                                  ((rImg_3_2-rMod_4_2)*(rImg_3_2-rMod_4_2)) + ((rImg_3_3-rMod_4_3)*(rImg_3_3-rMod_4_3)) +
                                  ((rImg_3_4-rMod_4_4)*(rImg_3_4-rMod_4_4)) + ((rImg_3_5-rMod_4_5)*(rImg_3_5-rMod_4_5)) +
                                  ((rImg_3_6-rMod_4_6)*(rImg_3_6-rMod_4_6)) + ((rImg_3_7-rMod_4_7)*(rImg_3_7-rMod_4_7)) +
                                  ((rImg_3_8-rMod_4_8)*(rImg_3_8-rMod_4_8)) + ((rImg_3_9-rMod_4_9)*(rImg_3_9-rMod_4_9)) +
                                  ((rImg_3_10-rMod_4_10)*(rImg_3_10-rMod_4_10)) + ((rImg_3_11-rMod_4_11)*(rImg_3_11-rMod_4_11)) +
                                  ((rImg_3_12-rMod_4_12)*(rImg_3_12-rMod_4_12)) + ((rImg_3_13-rMod_4_13)*(rImg_3_13-rMod_4_13)) +
                                  ((rImg_3_14-rMod_4_14)*(rImg_3_14-rMod_4_14)) + ((rImg_3_15-rMod_4_15)*(rImg_3_15-rMod_4_15));
                dist35 = dist35 + ((rImg_3_0-rMod_5_0)*(rImg_3_0-rMod_5_0)) + ((rImg_3_1-rMod_5_1)*(rImg_3_1-rMod_5_1)) +
                                  ((rImg_3_2-rMod_5_2)*(rImg_3_2-rMod_5_2)) + ((rImg_3_3-rMod_5_3)*(rImg_3_3-rMod_5_3)) +
                                  ((rImg_3_4-rMod_5_4)*(rImg_3_4-rMod_5_4)) + ((rImg_3_5-rMod_5_5)*(rImg_3_5-rMod_5_5)) +
                                  ((rImg_3_6-rMod_5_6)*(rImg_3_6-rMod_5_6)) + ((rImg_3_7-rMod_5_7)*(rImg_3_7-rMod_5_7)) +
                                  ((rImg_3_8-rMod_5_8)*(rImg_3_8-rMod_5_8)) + ((rImg_3_9-rMod_5_9)*(rImg_3_9-rMod_5_9)) +
                                  ((rImg_3_10-rMod_5_10)*(rImg_3_10-rMod_5_10)) + ((rImg_3_11-rMod_5_11)*(rImg_3_11-rMod_5_11)) +
                                  ((rImg_3_12-rMod_5_12)*(rImg_3_12-rMod_5_12)) + ((rImg_3_13-rMod_5_13)*(rImg_3_13-rMod_5_13)) +
                                  ((rImg_3_14-rMod_5_14)*(rImg_3_14-rMod_5_14)) + ((rImg_3_15-rMod_5_15)*(rImg_3_15-rMod_5_15));
                //5th Row
                dist40 = dist40 + ((rImg_4_0-rMod_0_0)*(rImg_4_0-rMod_0_0)) + ((rImg_4_1-rMod_0_1)*(rImg_4_1-rMod_0_1)) +
                                  ((rImg_4_2-rMod_0_2)*(rImg_4_2-rMod_0_2)) + ((rImg_4_3-rMod_0_3)*(rImg_4_3-rMod_0_3)) +
                                  ((rImg_4_4-rMod_0_4)*(rImg_4_4-rMod_0_4)) + ((rImg_4_5-rMod_0_5)*(rImg_4_5-rMod_0_5)) +
                                  ((rImg_4_6-rMod_0_6)*(rImg_4_6-rMod_0_6)) + ((rImg_4_7-rMod_0_7)*(rImg_4_7-rMod_0_7)) +
                                  ((rImg_4_8-rMod_0_8)*(rImg_4_8-rMod_0_8)) + ((rImg_4_9-rMod_0_9)*(rImg_4_9-rMod_0_9)) +
                                  ((rImg_4_10-rMod_0_10)*(rImg_4_10-rMod_0_10)) + ((rImg_4_11-rMod_0_11)*(rImg_4_11-rMod_0_11)) +
                                  ((rImg_4_12-rMod_0_12)*(rImg_4_12-rMod_0_12)) + ((rImg_4_13-rMod_0_13)*(rImg_4_13-rMod_0_13)) +
                                  ((rImg_4_14-rMod_0_14)*(rImg_4_14-rMod_0_14)) + ((rImg_4_15-rMod_0_15)*(rImg_4_15-rMod_0_15));
                dist41 = dist41 + ((rImg_4_0-rMod_1_0)*(rImg_4_0-rMod_1_0)) + ((rImg_4_1-rMod_1_1)*(rImg_4_1-rMod_1_1)) +
                                  ((rImg_4_2-rMod_1_2)*(rImg_4_2-rMod_1_2)) + ((rImg_4_3-rMod_1_3)*(rImg_4_3-rMod_1_3)) +
                                  ((rImg_4_4-rMod_1_4)*(rImg_4_4-rMod_1_4)) + ((rImg_4_5-rMod_1_5)*(rImg_4_5-rMod_1_5)) +
                                  ((rImg_4_6-rMod_1_6)*(rImg_4_6-rMod_1_6)) + ((rImg_4_7-rMod_1_7)*(rImg_4_7-rMod_1_7)) +
                                  ((rImg_4_8-rMod_1_8)*(rImg_4_8-rMod_1_8)) + ((rImg_4_9-rMod_1_9)*(rImg_4_9-rMod_1_9)) +
                                  ((rImg_4_10-rMod_1_10)*(rImg_4_10-rMod_1_10)) + ((rImg_4_11-rMod_1_11)*(rImg_4_11-rMod_1_11)) +
                                  ((rImg_4_12-rMod_1_12)*(rImg_4_12-rMod_1_12)) + ((rImg_4_13-rMod_1_13)*(rImg_4_13-rMod_1_13)) +
                                  ((rImg_4_14-rMod_1_14)*(rImg_4_14-rMod_1_14)) + ((rImg_4_15-rMod_1_15)*(rImg_4_15-rMod_1_15));
                dist42 = dist42 + ((rImg_4_0-rMod_2_0)*(rImg_4_0-rMod_2_0)) + ((rImg_4_1-rMod_2_1)*(rImg_4_1-rMod_2_1)) +
                                  ((rImg_4_2-rMod_2_2)*(rImg_4_2-rMod_2_2)) + ((rImg_4_3-rMod_2_3)*(rImg_4_3-rMod_2_3)) +
                                  ((rImg_4_4-rMod_2_4)*(rImg_4_4-rMod_2_4)) + ((rImg_4_5-rMod_2_5)*(rImg_4_5-rMod_2_5)) +
                                  ((rImg_4_6-rMod_2_6)*(rImg_4_6-rMod_2_6)) + ((rImg_4_7-rMod_2_7)*(rImg_4_7-rMod_2_7)) +
                                  ((rImg_4_8-rMod_2_8)*(rImg_4_8-rMod_2_8)) + ((rImg_4_9-rMod_2_9)*(rImg_4_9-rMod_2_9)) +
                                  ((rImg_4_10-rMod_2_10)*(rImg_4_10-rMod_2_10)) + ((rImg_4_11-rMod_2_11)*(rImg_4_11-rMod_2_11)) +
                                  ((rImg_4_12-rMod_2_12)*(rImg_4_12-rMod_2_12)) + ((rImg_4_13-rMod_2_13)*(rImg_4_13-rMod_2_13)) +
                                  ((rImg_4_14-rMod_2_14)*(rImg_4_14-rMod_2_14)) + ((rImg_4_15-rMod_2_15)*(rImg_4_15-rMod_2_15));
                dist43 = dist43 + ((rImg_4_0-rMod_3_0)*(rImg_4_0-rMod_3_0)) + ((rImg_4_1-rMod_3_1)*(rImg_4_1-rMod_3_1)) +
                                  ((rImg_4_2-rMod_3_2)*(rImg_4_2-rMod_3_2)) + ((rImg_4_3-rMod_3_3)*(rImg_4_3-rMod_3_3)) +
                                  ((rImg_4_4-rMod_3_4)*(rImg_4_4-rMod_3_4)) + ((rImg_4_5-rMod_3_5)*(rImg_4_5-rMod_3_5)) +
                                  ((rImg_4_6-rMod_3_6)*(rImg_4_6-rMod_3_6)) + ((rImg_4_7-rMod_3_7)*(rImg_4_7-rMod_3_7)) +
                                  ((rImg_4_8-rMod_3_8)*(rImg_4_8-rMod_3_8)) + ((rImg_4_9-rMod_3_9)*(rImg_4_9-rMod_3_9)) +
                                  ((rImg_4_10-rMod_3_10)*(rImg_4_10-rMod_3_10)) + ((rImg_4_11-rMod_3_11)*(rImg_4_11-rMod_3_11)) +
                                  ((rImg_4_12-rMod_3_12)*(rImg_4_12-rMod_3_12)) + ((rImg_4_13-rMod_3_13)*(rImg_4_13-rMod_3_13)) +
                                  ((rImg_4_14-rMod_3_14)*(rImg_4_14-rMod_3_14)) + ((rImg_4_15-rMod_3_15)*(rImg_4_15-rMod_3_15));
                dist44 = dist44 + ((rImg_4_0-rMod_4_0)*(rImg_4_0-rMod_4_0)) + ((rImg_4_1-rMod_4_1)*(rImg_4_1-rMod_4_1)) +
                                  ((rImg_4_2-rMod_4_2)*(rImg_4_2-rMod_4_2)) + ((rImg_4_3-rMod_4_3)*(rImg_4_3-rMod_4_3)) +
                                  ((rImg_4_4-rMod_4_4)*(rImg_4_4-rMod_4_4)) + ((rImg_4_5-rMod_4_5)*(rImg_4_5-rMod_4_5)) +
                                  ((rImg_4_6-rMod_4_6)*(rImg_4_6-rMod_4_6)) + ((rImg_4_7-rMod_4_7)*(rImg_4_7-rMod_4_7)) +
                                  ((rImg_4_8-rMod_4_8)*(rImg_4_8-rMod_4_8)) + ((rImg_4_9-rMod_4_9)*(rImg_4_9-rMod_4_9)) +
                                  ((rImg_4_10-rMod_4_10)*(rImg_4_10-rMod_4_10)) + ((rImg_4_11-rMod_4_11)*(rImg_4_11-rMod_4_11)) +
                                  ((rImg_4_12-rMod_4_12)*(rImg_4_12-rMod_4_12)) + ((rImg_4_13-rMod_4_13)*(rImg_4_13-rMod_4_13)) +
                                  ((rImg_4_14-rMod_4_14)*(rImg_4_14-rMod_4_14)) + ((rImg_4_15-rMod_4_15)*(rImg_4_15-rMod_4_15));
                dist45 = dist45 + ((rImg_4_0-rMod_5_0)*(rImg_4_0-rMod_5_0)) + ((rImg_4_1-rMod_5_1)*(rImg_4_1-rMod_5_1)) +
                                  ((rImg_4_2-rMod_5_2)*(rImg_4_2-rMod_5_2)) + ((rImg_4_3-rMod_5_3)*(rImg_4_3-rMod_5_3)) +
                                  ((rImg_4_4-rMod_5_4)*(rImg_4_4-rMod_5_4)) + ((rImg_4_5-rMod_5_5)*(rImg_4_5-rMod_5_5)) +
                                  ((rImg_4_6-rMod_5_6)*(rImg_4_6-rMod_5_6)) + ((rImg_4_7-rMod_5_7)*(rImg_4_7-rMod_5_7)) +
                                  ((rImg_4_8-rMod_5_8)*(rImg_4_8-rMod_5_8)) + ((rImg_4_9-rMod_5_9)*(rImg_4_9-rMod_5_9)) +
                                  ((rImg_4_10-rMod_5_10)*(rImg_4_10-rMod_5_10)) + ((rImg_4_11-rMod_5_11)*(rImg_4_11-rMod_5_11)) +
                                  ((rImg_4_12-rMod_5_12)*(rImg_4_12-rMod_5_12)) + ((rImg_4_13-rMod_5_13)*(rImg_4_13-rMod_5_13)) +
                                  ((rImg_4_14-rMod_5_14)*(rImg_4_14-rMod_5_14)) + ((rImg_4_15-rMod_5_15)*(rImg_4_15-rMod_5_15));
                //6th Row
                dist50 = dist50 + ((rImg_5_0-rMod_0_0)*(rImg_5_0-rMod_0_0)) + ((rImg_5_1-rMod_0_1)*(rImg_5_1-rMod_0_1)) +
                                  ((rImg_5_2-rMod_0_2)*(rImg_5_2-rMod_0_2)) + ((rImg_5_3-rMod_0_3)*(rImg_5_3-rMod_0_3)) +
                                  ((rImg_5_4-rMod_0_4)*(rImg_5_4-rMod_0_4)) + ((rImg_5_5-rMod_0_5)*(rImg_5_5-rMod_0_5)) +
                                  ((rImg_5_6-rMod_0_6)*(rImg_5_6-rMod_0_6)) + ((rImg_5_7-rMod_0_7)*(rImg_5_7-rMod_0_7)) +
                                  ((rImg_5_8-rMod_0_8)*(rImg_5_8-rMod_0_8)) + ((rImg_5_9-rMod_0_9)*(rImg_5_9-rMod_0_9)) +
                                  ((rImg_5_10-rMod_0_10)*(rImg_5_10-rMod_0_10)) + ((rImg_5_11-rMod_0_11)*(rImg_5_11-rMod_0_11)) +
                                  ((rImg_5_12-rMod_0_12)*(rImg_5_12-rMod_0_12)) + ((rImg_5_13-rMod_0_13)*(rImg_5_13-rMod_0_13)) +
                                  ((rImg_5_14-rMod_0_14)*(rImg_5_14-rMod_0_14)) + ((rImg_5_15-rMod_0_15)*(rImg_5_15-rMod_0_15));
                dist51 = dist51 + ((rImg_5_0-rMod_1_0)*(rImg_5_0-rMod_1_0)) + ((rImg_5_1-rMod_1_1)*(rImg_5_1-rMod_1_1)) +
                                  ((rImg_5_2-rMod_1_2)*(rImg_5_2-rMod_1_2)) + ((rImg_5_3-rMod_1_3)*(rImg_5_3-rMod_1_3)) +
                                  ((rImg_5_4-rMod_1_4)*(rImg_5_4-rMod_1_4)) + ((rImg_5_5-rMod_1_5)*(rImg_5_5-rMod_1_5)) +
                                  ((rImg_5_6-rMod_1_6)*(rImg_5_6-rMod_1_6)) + ((rImg_5_7-rMod_1_7)*(rImg_5_7-rMod_1_7)) +
                                  ((rImg_5_8-rMod_1_8)*(rImg_5_8-rMod_1_8)) + ((rImg_5_9-rMod_1_9)*(rImg_5_9-rMod_1_9)) +
                                  ((rImg_5_10-rMod_1_10)*(rImg_5_10-rMod_1_10)) + ((rImg_5_11-rMod_1_11)*(rImg_5_11-rMod_1_11)) +
                                  ((rImg_5_12-rMod_1_12)*(rImg_5_12-rMod_1_12)) + ((rImg_5_13-rMod_1_13)*(rImg_5_13-rMod_1_13)) +
                                  ((rImg_5_14-rMod_1_14)*(rImg_5_14-rMod_1_14)) + ((rImg_5_15-rMod_1_15)*(rImg_5_15-rMod_1_15));
                dist52 = dist52 + ((rImg_5_0-rMod_2_0)*(rImg_5_0-rMod_2_0)) + ((rImg_5_1-rMod_2_1)*(rImg_5_1-rMod_2_1)) +
                                  ((rImg_5_2-rMod_2_2)*(rImg_5_2-rMod_2_2)) + ((rImg_5_3-rMod_2_3)*(rImg_5_3-rMod_2_3)) +
                                  ((rImg_5_4-rMod_2_4)*(rImg_5_4-rMod_2_4)) + ((rImg_5_5-rMod_2_5)*(rImg_5_5-rMod_2_5)) +
                                  ((rImg_5_6-rMod_2_6)*(rImg_5_6-rMod_2_6)) + ((rImg_5_7-rMod_2_7)*(rImg_5_7-rMod_2_7)) +
                                  ((rImg_5_8-rMod_2_8)*(rImg_5_8-rMod_2_8)) + ((rImg_5_9-rMod_2_9)*(rImg_5_9-rMod_2_9)) +
                                  ((rImg_5_10-rMod_2_10)*(rImg_5_10-rMod_2_10)) + ((rImg_5_11-rMod_2_11)*(rImg_5_11-rMod_2_11)) +
                                  ((rImg_5_12-rMod_2_12)*(rImg_5_12-rMod_2_12)) + ((rImg_5_13-rMod_2_13)*(rImg_5_13-rMod_2_13)) +
                                  ((rImg_5_14-rMod_2_14)*(rImg_5_14-rMod_2_14)) + ((rImg_5_15-rMod_2_15)*(rImg_5_15-rMod_2_15));
                dist53 = dist53 + ((rImg_5_0-rMod_3_0)*(rImg_5_0-rMod_3_0)) + ((rImg_5_1-rMod_3_1)*(rImg_5_1-rMod_3_1)) +
                                  ((rImg_5_2-rMod_3_2)*(rImg_5_2-rMod_3_2)) + ((rImg_5_3-rMod_3_3)*(rImg_5_3-rMod_3_3)) +
                                  ((rImg_5_4-rMod_3_4)*(rImg_5_4-rMod_3_4)) + ((rImg_5_5-rMod_3_5)*(rImg_5_5-rMod_3_5)) +
                                  ((rImg_5_6-rMod_3_6)*(rImg_5_6-rMod_3_6)) + ((rImg_5_7-rMod_3_7)*(rImg_5_7-rMod_3_7)) +
                                  ((rImg_5_8-rMod_3_8)*(rImg_5_8-rMod_3_8)) + ((rImg_5_9-rMod_3_9)*(rImg_5_9-rMod_3_9)) +
                                  ((rImg_5_10-rMod_3_10)*(rImg_5_10-rMod_3_10)) + ((rImg_5_11-rMod_3_11)*(rImg_5_11-rMod_3_11)) +
                                  ((rImg_5_12-rMod_3_12)*(rImg_5_12-rMod_3_12)) + ((rImg_5_13-rMod_3_13)*(rImg_5_13-rMod_3_13)) +
                                  ((rImg_5_14-rMod_3_14)*(rImg_5_14-rMod_3_14)) + ((rImg_5_15-rMod_3_15)*(rImg_5_15-rMod_3_15));
                dist54 = dist54 + ((rImg_5_0-rMod_4_0)*(rImg_5_0-rMod_4_0)) + ((rImg_5_1-rMod_4_1)*(rImg_5_1-rMod_4_1)) +
                                  ((rImg_5_2-rMod_4_2)*(rImg_5_2-rMod_4_2)) + ((rImg_5_3-rMod_4_3)*(rImg_5_3-rMod_4_3)) +
                                  ((rImg_5_4-rMod_4_4)*(rImg_5_4-rMod_4_4)) + ((rImg_5_5-rMod_4_5)*(rImg_5_5-rMod_4_5)) +
                                  ((rImg_5_6-rMod_4_6)*(rImg_5_6-rMod_4_6)) + ((rImg_5_7-rMod_4_7)*(rImg_5_7-rMod_4_7)) +
                                  ((rImg_5_8-rMod_4_8)*(rImg_5_8-rMod_4_8)) + ((rImg_5_9-rMod_4_9)*(rImg_5_9-rMod_4_9)) +
                                  ((rImg_5_10-rMod_4_10)*(rImg_5_10-rMod_4_10)) + ((rImg_5_11-rMod_4_11)*(rImg_5_11-rMod_4_11)) +
                                  ((rImg_5_12-rMod_4_12)*(rImg_5_12-rMod_4_12)) + ((rImg_5_13-rMod_4_13)*(rImg_5_13-rMod_4_13)) +
                                  ((rImg_5_14-rMod_4_14)*(rImg_5_14-rMod_4_14)) + ((rImg_5_15-rMod_4_15)*(rImg_5_15-rMod_4_15));
                dist55 = dist55 + ((rImg_5_0-rMod_5_0)*(rImg_5_0-rMod_5_0)) + ((rImg_5_1-rMod_5_1)*(rImg_5_1-rMod_5_1)) +
                                  ((rImg_5_2-rMod_5_2)*(rImg_5_2-rMod_5_2)) + ((rImg_5_3-rMod_5_3)*(rImg_5_3-rMod_5_3)) +
                                  ((rImg_5_4-rMod_5_4)*(rImg_5_4-rMod_5_4)) + ((rImg_5_5-rMod_5_5)*(rImg_5_5-rMod_5_5)) +
                                  ((rImg_5_6-rMod_5_6)*(rImg_5_6-rMod_5_6)) + ((rImg_5_7-rMod_5_7)*(rImg_5_7-rMod_5_7)) +
                                  ((rImg_5_8-rMod_5_8)*(rImg_5_8-rMod_5_8)) + ((rImg_5_9-rMod_5_9)*(rImg_5_9-rMod_5_9)) +
                                  ((rImg_5_10-rMod_5_10)*(rImg_5_10-rMod_5_10)) + ((rImg_5_11-rMod_5_11)*(rImg_5_11-rMod_5_11)) +
                                  ((rImg_5_12-rMod_5_12)*(rImg_5_12-rMod_5_12)) + ((rImg_5_13-rMod_5_13)*(rImg_5_13-rMod_5_13)) +
                                  ((rImg_5_14-rMod_5_14)*(rImg_5_14-rMod_5_14)) + ((rImg_5_15-rMod_5_15)*(rImg_5_15-rMod_5_15));
                __syncthreads();
        }
        //1st Row
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist00;
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist01;
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist02;
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist03;
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist04;
        if(((valy+(0*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(0*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist05;
        //2nd Row
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist10;
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist11;
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist12;
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist13;
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist14;
        if(((valy+(1*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(1*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist15;
        //3rd Row
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist20;
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist21;
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist22;
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist23;
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist24;
        if(((valy+(2*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(2*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist25;
        //4th Row
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist30;
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist31;
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist32;
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist33;
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist34;
        if(((valy+(3*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(3*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist35;
        //5th Row
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist40;
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist41;
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist42;
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist43;
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist44;
        if(((valy+(4*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(4*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist45;
        //6th Row
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(0*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(0*TILE)+txId]=dist50;
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(1*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(1*TILE)+txId]=dist51;
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(2*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(2*TILE)+txId]=dist52;
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(3*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(3*TILE)+txId]=dist53;
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(4*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(4*TILE)+txId]=dist54;
        if(((valy+(5*TILE)+tyId)<numDataImgs)&&((valx+(5*TILE)+txId)<numRefImgs)) euDist[(valy*numRefImgs)+(5*TILE*numRefImgs)+(tyId*numRefImgs)+valx+(5*TILE)+txId]=dist55;
}

//CUDA kernel used to identify maximum value and index of the value in a vector.
__device__ int maxDistId(float *dist, int totalNN)
{
        float maxDist; int maxId;
        maxDist = dist[0]; maxId = 0;
        for(int i=1;i<totalNN;i++){
                if(dist[i]>maxDist){
                        maxDist = dist[i]; maxId = i;
                }
        }
        return maxId;
}

//CUDA kernel used to compute k-Neareast Neighbor.
__global__ void computeKNeighbors(dtype *dEuDist, dtype *dKDist, int *dKIndex, int numDataImgs, int numRefImgs, int totalNN)
{
        int imgId = (blockIdx.x*blockDim.x)+threadIdx.x;
        float maxDist; int id;

        if(imgId<numDataImgs){
                //Copy the distances and indexes of the first totalNN elements
                for(int j=0;j<totalNN;j++){
                        dKDist[(imgId*totalNN)+j] = dEuDist[(imgId*numRefImgs)+j];
                        dKIndex[(imgId*totalNN)+j] = j;
                }

                //Find the maximum in dKDist and its respective dKIndex
                id = maxDistId(&dKDist[imgId*totalNN],totalNN);
                maxDist = dKDist[(imgId*totalNN)+id];
                for(int j=totalNN;j<numRefImgs;j++){
                        if(dEuDist[(imgId*numRefImgs)+j]<maxDist){
                                dKDist[(imgId*totalNN)+id] = dEuDist[(imgId*numRefImgs)+j]; dKIndex[(imgId*totalNN)+id] = j;
                                //Find the maximum in dKDist and its respective dKIndex
                                id = maxDistId(&dKDist[imgId*totalNN],totalNN);
                                maxDist = dKDist[(imgId*totalNN)+id];
                        }
                }
        }
}

// dHeapify is a CUDA version of heapify code from GeeksforGeeks
// https://www.geeksforgeeks.org/heap-sort/
__device__ void dHeapify(float *arr, int *idx, int n, int i)
{
        int largest = i; // Initialize largest as root
        int l = 2*i + 1; // left = 2*i + 1
        int r = 2*i + 2; // right = 2*i + 2
        float tmpF; int tmpI;

        // If left child is larger than root
        if (l < n && arr[l] > arr[largest]) largest = l;

        // If right child is larger than largest so far
        if (r < n && arr[r] > arr[largest]) largest = r;

        // If largest is not root
        if (largest != i){
                //swap(arr[i], arr[largest]);
                tmpF = arr[i]; arr[i] = arr[largest]; arr[largest] = tmpF;
                tmpI = idx[i]; idx[i] = idx[largest]; idx[largest] = tmpI;

                // Recursively heapify the affected sub-tree
                dHeapify(arr, idx, n, largest);
        }
}

// dHeapify is a CUDA version of heapify code from GeeksforGeeks
// https://www.geeksforgeeks.org/heap-sort/
__global__ void computeKSort(float *dKDist, int *dKIndex, int numDataImgs, int totalNN)
{
        int imgId = (blockIdx.x*blockDim.x)+threadIdx.x;

        float tmpF; int tmpI;
        if(imgId<numDataImgs){
                //Build heap (rearrange array)
                for (int i = totalNN / 2 - 1; i >= 0; i--) dHeapify(&dKDist[imgId*totalNN], &dKIndex[imgId*totalNN], totalNN, i);

                // One by one extract an element from heap
                for (int i=totalNN-1; i>=0; i--){
                        // Move current root to end
                        //swap(arr[0], arr[i]);
                        tmpF = dKDist[(imgId*totalNN)+0]; dKDist[(imgId*totalNN)+0] = dKDist[(imgId*totalNN)+i]; dKDist[(imgId*totalNN)+i] = tmpF;
                        tmpI = dKIndex[(imgId*totalNN)+0]; dKIndex[(imgId*totalNN)+0] = dKIndex[(imgId*totalNN)+i]; dKIndex[(imgId*totalNN)+i] = tmpI;

                        // call max heapify on the reduced heap
                        dHeapify(&dKDist[imgId*totalNN], &dKIndex[imgId*totalNN], i, 0);
                }
        }
}

//Function used to invoke CUDA kernels.
py::array_t<dtype> cudaComputeEuclideanDistance(py::array_t<dtype> dataImgs, py::array_t<dtype> refImgs, int numDataImgs, int numRefImgs, int totalPixels, int deviceId)
{
	cout<<"[CUDA] In CUDA Euclidean distance function."<<endl;
	cout<<"[CUDA] Using deviceId: " << deviceId <<endl;
	py:: buffer_info dataBuf = dataImgs.request();
	py:: buffer_info refBuf = refImgs.request();
	cout<<"[CUDA] Dimension of dataImgs: "<<dataBuf.ndim<<" and refImgs: "<<refBuf.ndim<<endl;
	cout<<"[CUDA] Size of dataImgs: "<<dataBuf.size<<" and refImgs: "<<refBuf.size<<endl;

    cudaSetDevice(deviceId);

	//Declare and allocate device variables
	dtype *dDataImgs, *dRefImgs, *dEuDist;
        cudaMalloc((void **)&dDataImgs, numDataImgs*totalPixels*sizeof(dtype));
        cudaMalloc((void **)&dRefImgs, numRefImgs*totalPixels*sizeof(dtype));
        cudaMalloc((void **)&dEuDist, numDataImgs*numRefImgs*sizeof(dtype));
        //cudaMalloc((void **)&dKDist, numDataImgs*totalNN*sizeof(float));
        //cudaMalloc((void **)&dKIndex, numDataImgs*totalNN*sizeof(int));

	//Allocate euDist 
	py::array_t<dtype> euDist = py::array_t<dtype>(numDataImgs*numRefImgs);
	//py::array_t<dtype> kDist = py::array_t<dtype>(numDataImgs*sizeof(dtype));
	py::buffer_info euDistBuf = euDist.request();
	//py::buffer_info kDistBuf = kDist.request();

	//Obtain numpy data pointer
	dtype* dataPtr = reinterpret_cast<dtype*>(dataBuf.ptr);
	dtype* refPtr  = reinterpret_cast<dtype*>(refBuf.ptr);
	dtype* euDistPtr = reinterpret_cast<dtype*>(euDistBuf.ptr);
	//dtype* kDistPtr = (dtype*)kDistBuf.ptr;
	//int* kIndexPtr = (int*)kIndexBuf.ptr;

	//cout<<"[CUDA] Before dataPtr[0]: "<<dataPtr[0]<<", refPtr[0]: "<<refPtr[0]<<", and euDistPtr[0]: "<<euDistPtr[0]<<endl;

	auto start = high_resolution_clock::now();
	//Copy host to device
	cout<<"[CUDA] Copy data from host to device."<<endl;
	cudaMemcpy(dDataImgs, dataPtr, numDataImgs*totalPixels*sizeof(dtype), cudaMemcpyHostToDevice);
	cudaMemcpy(dRefImgs, refPtr, numRefImgs*totalPixels*sizeof(dtype), cudaMemcpyHostToDevice);
	//cudaMemcpy(dEuDist, euDistPtr, numDataImgs*numRefImgs*sizeof(dtype), cudaMemcpyHostToDevice);

	//Block and Grid Configuration
        int blockx = TILE;
        int blocky = TILE;
        dim3 block(blockx, blocky);
        int xImgPerBlock = TILE*imgPerThread; int yImgPerBlock = TILE*imgPerThread;
        dim3 grid((numRefImgs + xImgPerBlock - 1) / xImgPerBlock, (numDataImgs + yImgPerBlock - 1) / yImgPerBlock);

	//Kernel Call
        computeEuDistSharReg<<<grid, block>>>(dDataImgs,dRefImgs,dEuDist,numDataImgs,numRefImgs,totalPixels);
        cudaDeviceSynchronize();

	//Copy device to host
	cout<<"[CUDA] Copy data from device to host."<<endl;
	cudaMemcpy(euDistPtr,dEuDist,numDataImgs*numRefImgs*sizeof(float),cudaMemcpyDeviceToHost);

	//cout<<"[CUDA] euDistPtr values from kernel:"<<endl;
	//for(int i=0;i<100;i++) cout<<euDistPtr[i]<<"\t";
	//cout<<endl;
	//for(int i=0;i<100;i++) euDistPtr[i]=33;
	//cout<<"[CUDA] euDistPtr values from initialization:"<<endl;
        //for(int i=0;i<100;i++) cout<<euDistPtr[i]<<"\t";
        //cout<<endl;

	//Free memory
	cudaFree(dDataImgs); cudaFree(dRefImgs); cudaFree(dEuDist);
        //cudaFree(dKDist); cudaFree(dKIndex);	
	return euDist;
}

py::array_t<int> cudaComputeHeapSort(py::array_t<dtype> euDist, int numDataImgs, int numRefImgs, long int totalPixels, int totalNN, int deviceId)
{
	    cout<<"[CUDA] In CUDA Heap sort function."<<endl;
	    cout<<"[CUDA] Using deviceId: " << deviceId <<endl;
        py:: buffer_info euDistBuf = euDist.request();
        cout<<"[CUDA] Dimension of euDist: "<<euDistBuf.ndim<<endl;
        cout<<"[CUDA] Size of euDist: "<<euDistBuf.size<<endl;

        cudaSetDevice(deviceId);

        //Declare and allocate device variables
        dtype *dEuDist, *dKDist; int *dKIndex;
        cudaMalloc((void **)&dEuDist, numDataImgs*numRefImgs*sizeof(dtype));
        cudaMalloc((void **)&dKDist, numDataImgs*totalNN*sizeof(dtype));
        cudaMalloc((void **)&dKIndex, numDataImgs*totalNN*sizeof(int));

        //Allocate euDist
        py::array_t<int> kIndex = py::array_t<int>(numDataImgs*totalNN);
        py::buffer_info kIndexBuf = kIndex.request();
	cout<<"[CUDA] Dimension of kIndex: "<<kIndexBuf.ndim<<endl;
        cout<<"[CUDA] Size of kIndex: "<<kIndexBuf.size<<endl;

        //Obtain numpy data pointer
        dtype* euDistPtr = reinterpret_cast<dtype*>(euDistBuf.ptr);
	int* kIndexPtr = reinterpret_cast<int*>(kIndexBuf.ptr);

	//Copy Euclidean distance to device
	cudaMemcpy(dEuDist, euDistPtr, numDataImgs*numRefImgs*sizeof(float), cudaMemcpyHostToDevice);

	//Block and Grid Configuration
        int blockx = 32;
        int blocky = 1;
        dim3 block(blockx, blocky);
        dim3 grid((numDataImgs + blockx - 1) / blockx, 1);

	//Kernel Call
        computeKNeighbors<<<grid, block>>>(dEuDist,dKDist,dKIndex,numDataImgs,numRefImgs,totalNN);
        cudaDeviceSynchronize();
        computeKSort<<<grid, block>>>(dKDist,dKIndex,numDataImgs,totalNN);
        cudaDeviceSynchronize();

	//Copy results from device to host
        cudaMemcpy(kIndexPtr,dKIndex,numDataImgs*totalNN*sizeof(int),cudaMemcpyDeviceToHost);

	//Free memory
        cudaFree(dEuDist); cudaFree(dKDist); cudaFree(dKIndex);

        return kIndex;
}

PYBIND11_MODULE(pyCudaKNearestNeighbors, cudaBind)
{
	cudaBind.doc() = "Compute K-Nearest Neighbots between data and reference images.";
	cudaBind.def("cudaEuclideanDistance", cudaComputeEuclideanDistance);
	cudaBind.def("cudaHeapSort", cudaComputeHeapSort);
}
