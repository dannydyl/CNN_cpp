#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <cstdlib>   // for rand(), srand()
#include <ctime>     // for time()
#include <fstream>
#include <vector>
#include <complex>
#include <math.h>
#include <cassert>

using namespace std;



// allocate memory for a 1d vector
void alloc1d(vector<double>& tn1d, int s1) {
    // allocate memory for a 1d tensor tn1d of size s1
    tn1d.resize(s1);
}

// read data for a 1d tensor
void read1d(vector<double>& tn1d, int s1, istream& input_file) {
    for (int i1 = 0; i1 < s1; i1++)
    {
        input_file >> tn1d[i1];
    }
}
// print data for a 1d tensor
void print1d(vector<double>& tn1d, int s1) {
    cout << endl << endl;
    for (int i1 = 0; i1 < s1; i1++)
    {
        cout << tn1d[i1] << "   ";
    }
    cout << endl << endl;
}

// allocate memory for a 3d tensor
void alloc3d(vector<vector<vector<double> > >& tn3d, int s1, int s2, int s3) {
    // allocate memory for a 3d tensor tn3d of size s1, s2, s3
    tn3d.resize(s1);
    for (int i1 = 0; i1 < s1; i1++) {
        tn3d[i1].resize(s2);
        for (int i2 = 0; i2 < s2; i2++) {
            tn3d[i1][i2].resize(s3);
        }
    }
}

// read data for a 3d tensor
void read3d(vector<vector<vector<double> > >& tn3d, int s1, int s2, int s3, istream& input_file) {
    for (int i3 = 0; i3 < s3; i3++) {
        for (int i2 = 0; i2 < s2; i2++) {
            for (int i1 = 0; i1 < s1; i1++)
            {
                input_file >> tn3d[i1][i2][i3];
            }
        }
    }
}

// print data for a 3d tensor
void print3d(vector<vector<vector<double> > >& tn3d, int s1, int s2, int s3) {
    cout << endl << endl;
    for (int i3 = 0; i3 < s3; i3++) {
        for (int i2 = 0; i2 < s2; i2++) {
            for (int i1 = 0; i1 < s1; i1++)
            {
                cout << tn3d[i1][i2][i3] << "  ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }
    cout << endl << endl;
}

// allocate memory for a 4d tensor tn4d of size s1, s2, s3, s4
void alloc4d(vector<vector<vector<vector<double> > > >& tn4d, int s1, int s2, int s3, int s4) {

    tn4d.resize(s1);

    for (int i1 = 0; i1 < s1; i1++) {
        tn4d[i1].resize(s2);
        for (int i2 = 0; i2 < s2; i2++) {
            tn4d[i1][i2].resize(s3);
            for (int i3 = 0; i3 < s3; i3++)
            {
                tn4d[i1][i2][i3].resize(s4);
            }
        }
    }
}

// read a 4d tensor tn4d of size s1, s2, s3, s4
void read4d(vector<vector<vector<vector<double> > > >& tn4d, int s1, int s2, int s3, int s4, istream& input_file) {

    for (int i4 = 0; i4 < s4; i4++) {
        for (int i3 = 0; i3 < s3; i3++) {
            for (int i2 = 0; i2 < s2; i2++) {
                for (int i1 = 0; i1 < s1; i1++) {
                    input_file >> tn4d[i1][i2][i3][i4];
                }
            }
        }
    }
}

// print a 4d tensor tn4d of size s1, s2, s3, s4
void print4d(vector<vector<vector<vector<double> > > >& tn4d, int s1, int s2, int s3, int s4) {
    //cout << endl << endl;
    for (int i4 = 0; i4 < s4; i4++) {
        for (int i3 = 0; i3 < s3; i3++) {
            for (int i2 = 0; i2 < s2; i2++) {
                for (int i1 = 0; i1 < s1; i1++) {
                    cout << tn4d[i1][i2][i3][i4] << "   ";
                }
                cout << endl;
            }
            cout << endl << endl;
        }
        cout << endl << endl;
    }
    cout << endl << endl;
}

// method 4 convolve D1 with D3 to get D2
void convolM4(vector<vector<vector<double> > >& D1, vector<vector<vector<vector<double> > > >& D3, vector<double>& D4, vector<vector<vector<double> > >& D2){
    // 	tn1 : D1 : 32x32x3,  tn2: D3 : 5x5x3x16,   tn3: D2: 32x32x3

    //  bias [16] :  D4

	// "Convolve" tn1 with tn2 with stride 2 to obtain tn3
	int stride = 1;
	int tn1s1 = 32, tn1s2 = 32, tn1s3 = 3;
	int tn2s1 = 5, tn2s2 = 5, tn2s3 = 3, tn2s4 = 16;
	int tn2s1by2 = tn2s1 / 2;	// 2
	int tn2s2by2 = tn2s2 / 2;	// 2
	int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;



	cout << "Output of Convolution layerM4" << endl;

	for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
		for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
			for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
				double tmpsum = 0.0;
				for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
					// note tn1s3=tn2s3
					for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
						for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
							if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) <tn1s1)) { // zero padding of tn1
								tmpsum += D3[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D1[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
							}
						}
					}
				}
				D2[tn3i1][tn3i2][tn3i3] = tmpsum  + D4[tn3i3];  // D2 = D1 * D3 + D4
				//cout << D2[tn3i1][tn3i2][tn3i3] << "  ";
			}
			cout << endl;
		}
		cout << endl << endl;
	}
}

void ReLuLayer3(vector<vector<vector<double> > >& D2, vector<vector<vector<double> > >& D5){
        
    for (int d = 0; d < 16; ++d) {
        for (int h = 0; h < 32; ++h) {
            for (int w = 0; w < 32; ++w) {
                D5[d][h][w] = max(0.0, D2[d][h][w]);
            }
        }
    }  
}

void maxPoolingLayer4(vector<vector<vector<double> > >& D5, vector<vector<vector<double> > >& D6){
    int m = 32, n = 32, k = 16;
    int i = 16, j = 16;
    // Input: D5: 32x32x16 say D5[m][n][k]
    // Output: D6: 16x16x16, say D6[i][j][k]
    int stride = 2;

    for(k=0;k<16;k++) {  // for each cross section k
        for(m=0, i=0;m<32;m += stride ,i++) { //for row m
            for(n=0, j=0;n<32;n +=stride, j++) { // for column n
                double max1 = max(D5[m][n][k], D5[m+1][n][k]);
                double max2 = max(D5[m][n+1][k], D5[m+1][n+1][k]);
                D6[i][j][k] = max(max1, max2);
            }
        }
    }
}

void convolM9(vector<vector<vector<double> > >& D6, vector<vector<vector<vector<double> > > >& D8, vector<double>& D9, vector<vector<vector<double> > >& D7){

	// "Convolve" tn1 with tn2 with stride 2 to obtain tn3
	int stride = 1;
	int tn1s1 = 16, tn1s2 = 16, tn1s3 = 16;
	int tn2s1 = 5, tn2s2 = 5, tn2s3 = 16, tn2s4 = 20;
	int tn2s1by2 = tn2s1 / 2;	// 2
	int tn2s2by2 = tn2s2 / 2;	// 2
	int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;



	cout << "Output of Convolution layerM9" << endl;

	for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
		for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
			for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
				double tmpsum = 0.0;
				for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
					// note tn1s3=tn2s3
					for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
						for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
							if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) <tn1s1)) { // zero padding of tn1
								tmpsum += D8[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D6[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
                                
							}
						}
					}
				}
				D7[tn3i1][tn3i2][tn3i3] = tmpsum  + D9[tn3i3];  // D2 = D1 * D3 + D4
				//cout << D7[tn3i1][tn3i2][tn3i3] << "  ";
                
			}
			cout << endl;
                
		}
		cout << endl << endl;
	}
    
}

void ReLuLayer6(vector<vector<vector<double> > >& D7, vector<vector<vector<double> > >& D10){
        
    for (int d = 0; d < 16; ++d) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 20; ++w) {
                D10[d][h][w] = max(0.0, D7[d][h][w]);
            }
        }
    }  
    
}

void maxPoolingLayer7(vector<vector<vector<double> > >& D10, vector<vector<vector<double> > >& D11){
    int m = 8, n = 8, k = 20;
    int i = 2, j = 2;
    // Input: D5: 32x32x16 say D5[m][n][k]
    // Output: D6: 16x16x16, say D6[i][j][k]
    int stride = 2;

    for(k=0;k<20;k++) {  // for each cross section k
        for(m=0, i=0;m<8;m += stride ,i++) { //for row m
            for(n=0, j=0;n<8;n +=stride, j++) { // for column n
                double max1 = max(D10[m][n][k], D10[m+1][n][k]);
                double max2 = max(D10[m][n+1][k], D10[m+1][n+1][k]);
                D11[i][j][k] = max(max1, max2);
            }
        }
    }
}

void convolM14(vector<vector<vector<double> > >& D11, vector<vector<vector<vector<double> > > >& D13, vector<double>& D14, vector<vector<vector<double> > >& D12){
    // "Convolve" tn1 with tn2 with stride 2 to obtain tn3
	int stride = 1;
	int tn1s1 = 8, tn1s2 = 8, tn1s3 = 20;
	int tn2s1 = 5, tn2s2 = 5, tn2s3 = 20, tn2s4 = 20;
	int tn2s1by2 = tn2s1 / 2;	// 2
	int tn2s2by2 = tn2s2 / 2;	// 2
	int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;

	cout << "Output of Convolution layerM14" << endl;

	for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
		for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
			for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
				double tmpsum = 0.0;
				for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
					// note tn1s3=tn2s3
					for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
						for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
							if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) <tn1s1)) { // zero padding of tn1
								tmpsum += D13[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D11[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
							}
						}
					}
				}
				D12[tn3i1][tn3i2][tn3i3] = tmpsum  + D14[tn3i3];  // D2 = D1 * D3 + D4
				//cout << D12[tn3i1][tn3i2][tn3i3] << "  ";
			}
			//cout << endl;
		}
		//cout << endl << endl;
	}
    
}

void ReLuLayer9(vector<vector<vector<double> > >& D12, vector<vector<vector<double> > >& D15){
    // Apply ReLU activation for Layer L9
    for (int d = 0; d < 8; ++d) {
        for (int h = 0; h < 8; ++h) {
            for (int w = 0; w < 20; ++w) {
                D15[d][h][w] = max(0.0, D12[d][h][w]);
            }
        }
    }
}

void maxPoolingLayer10(vector<vector<vector<double> > >& D15, vector<vector<vector<double> > >& D16){
    // Max pooling for Layer L10
    for(int k = 0; k < 20; k++) {
        for(int m = 0, i = 0; m < 4; m += 2, i++) {
            for(int n = 0, j = 0; n < 4; n += 2, j++) {
                double max1 = max(D15[m][n][k], D15[m+1][n][k]);
                double max2 = max(D15[m][n+1][k], D15[m+1][n+1][k]);
                D16[i][j][k] = max(max1, max2);
            }
        }
    }
}

void fullyConnectedLayer11(vector<vector<vector<double> > >& D16, vector<vector<vector<vector<double> > > >& D18, vector<double>& D19, vector<double>& D17) {
    // Initialize D17 with size 10
    //D17.resize(10, 0.0);

    // Compute the dot product of D16 with D18 and add the bias from D19
    for (int i = 0; i < 10; ++i) { // For each filter in D18
        double sum = 0.0;
        for (int d = 0; d < 20; ++d) { // Depth
            for (int h = 0; h < 4; ++h) { // Height
                for (int w = 0; w < 4; ++w) { // Width
                    sum += D16[h][w][d] * D18[h][w][d][i];
                }
            }
        }
        
        D17[i] = sum + D19[i];
 
    }
}

void softmaxLayer12(vector<double>& D17, vector<double>& D20) {
    D20.resize(10, 0.0);

    // Compute the sum of exponentials
    double sumOfExps = 0.0;
    for (int i = 0; i < D17.size(); ++i) {
        sumOfExps += exp(D17[i]);
    }

    // Apply softmax
    for (int i = 0; i < 10; ++i) {
        D20[i] = exp(D17[i]) / sumOfExps;
    }
}

int main(){

    ifstream input_file, input_image; 
    input_file.open("CNN_weights.txt");
    input_image.open("Test_image.txt");
    
    vector<double> D4, D9;
    vector<vector<vector<double> > > D1, D2, D5, D6, D7, D10, D11;
    vector<vector<vector<vector<double> > > > D3, D8;

    alloc3d(D1, 32, 32, 3); //D1 <- this should be image thats loaded
    read3d(D1, 32, 32, 3, input_image);

    cout << "Input file: " << endl;
    print3d(D1, 32, 32, 3);

    //cout << endl << "D3: 5x5x3x16";
    alloc4d(D3, 5, 5, 3, 16); //D3 <- in file
    read4d(D3, 5, 5, 3, 16, input_file); //D3
    //print4d(D3, 5, 5, 3, 16); //D3

    // D2 32x32x16
    // alloc3d(D2, 32, 32, 16); //D2
    // read3d(D2, 32, 32, 16, input_image);
    // print3d(D2, 32, 32, 16);

    // D4 16x1
    alloc1d(D4, 16); //D4 <- in file
    read1d(D4, 16, input_file); //
    print1d(D4, 16); //

    alloc3d(D2, 32, 32, 16); // allocating memory for D2
    convolM4(D1, D3, D4, D2);
    // print3d(D2, 32, 32, 3);
    
    alloc3d(D5, 32, 32, 16); //D5 
    ReLuLayer3(D2, D5); // Relu layer

    alloc3d(D6, 16, 16, 16); //D6

    maxPoolingLayer4(D5, D6); // Max pooling layer

    alloc3d(D7, 16, 16, 20); //D7
    
    alloc4d(D8, 5, 5, 16, 20); //D8
    read4d(D8, 5, 5, 16, 20, input_file);
    //D8 input file required

    alloc1d(D9, 20); //D9
    read1d(D9, 20, input_file);
    // D9 input file required
    convolM9(D6, D8, D9, D7); // Convolution layer
    
    alloc3d(D10, 16, 16, 20); //D10
    ReLuLayer6(D7, D10); // Relu layer

    alloc3d(D11, 8, 8, 20); //D11
    maxPoolingLayer7(D10, D11); // Max pooling layer


    // stage 4
    vector<vector<vector<double> > > D12, D15, D16;
    vector<vector<vector<vector<double> > > > D13;
    vector<double> D14;

    alloc3d(D12, 8, 8, 20); //D12
    alloc4d(D13, 5, 5, 20, 20); //D13
    read4d(D13, 5, 5, 20, 20, input_file);
    // D13 input file required
    alloc1d(D14, 20); //D14
    read1d(D14, 20, input_file);
    // D14 input file required
    alloc3d(D15, 8, 8, 20); //D15
    alloc3d(D16, 4, 4, 20); // D16

    convolM14(D11, D13, D14, D12);
    ReLuLayer9(D12, D15);
    maxPoolingLayer10(D15, D16);
    vector<double> D17, D19;
    vector<vector<vector<vector<double> > > > D18;

    alloc1d(D17, 10);
    alloc4d(D18, 4, 4, 20, 10);
    read4d(D18, 4, 4, 20, 10, input_file);
    // D18 input file required
    alloc1d(D19, 10);
    read1d(D19, 10, input_file);
    // D19 input file required


    // Fully connected layer
    fullyConnectedLayer11(D16, D18, D19, D17);
    for (int i = 0; i < D17.size(); ++i) {
        cout << D17[i] << " ";
    }
    cout << endl;
    // Softmax layer
    vector<double> D20;
    alloc1d(D20, 10);
    softmaxLayer12(D17, D20);

    // Print the output of the softmax layer (D20)
    cout << "Output of the softmax layer: " << endl;
    for (int i = 0; i < D20.size(); ++i) {
        cout << D20[i] << " ";
    }
    cout << endl;


    input_file.close();
    input_image.close();
    return 0;
}

#endif // MATRIX_H
