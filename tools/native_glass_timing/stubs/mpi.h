#ifndef GLASS_TIMING_MPI_STUB_H
#define GLASS_TIMING_MPI_STUB_H
typedef int MPI_Comm;
typedef struct { int unused; } MPI_Status;
#define MPI_INT 0
#define MPI_DOUBLE 0
#define MPI_COMM_WORLD 0
int MPI_Send();
int MPI_Recv();
#endif
