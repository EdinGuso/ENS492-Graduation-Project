max_CD = 0.4490;
min_CD = -0.0014;
max_CL = 0.1155;
min_CL = -1.8535;

CD_error = 10.38;
CL_error = 156.54;

new_CD_error = (CD_error * max_CD) / (max_CD - min_CD)
new_CL_error = (CL_error * max_CL) / (max_CL - min_CL)