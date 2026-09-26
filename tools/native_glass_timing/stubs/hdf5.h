#ifndef GLASS_TIMING_HDF5_STUB_H
#define GLASS_TIMING_HDF5_STUB_H
typedef long long hid_t;
typedef unsigned long long hsize_t;
typedef int herr_t;
#define H5F_ACC_RDONLY 0
#define H5P_DEFAULT 0
#define H5S_ALL 0
#define H5T_COMPOUND 0
#define H5T_IEEE_F64LE 0
#define H5T_NATIVE_DOUBLE 0
#define H5T_NATIVE_INT 0
#define HOFFSET(S, M) 0
hid_t H5Fopen(); hid_t H5Dopen(); hid_t H5Dget_space(); hid_t H5Dget_type();
int H5Sget_simple_extent_ndims(); int H5Sget_simple_extent_dims();
hid_t H5Tcreate(); herr_t H5Tinsert(); herr_t H5Dread();
herr_t H5Dclose(); herr_t H5Fclose(); herr_t H5Sclose(); herr_t H5Tclose();
#endif
