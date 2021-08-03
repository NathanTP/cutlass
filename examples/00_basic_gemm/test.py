import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import ctypes as ct

def getCubin():
    mod = cuda.module_from_file("kern.cubin")
    func = mod.get_function("_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")
    func.prepare("P")
    return func

libc = ct.cdll.LoadLibrary("lib/out.so")
getArg = libc.adaptSGEMMArgs
c_float_p = ct.POINTER(ct.c_float)
getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_float_p, ct.c_int,\
c_float_p, ct.c_int, ct.c_float, c_float_p, ct.c_int]
getArg.restypes = ct.c_void_p

# getConf = libc.getCudaConfig

f = getCubin()

np.random.seed(5)
a = np.random.rand(8, 8)
b = np.random.rand(8, 8)
c = np.zeros_like(a)

a_d = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_d, a)
b_d = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_d, b)
c_d = cuda.mem_alloc(b.nbytes)
len_d = cuda.mem_alloc(64)
cuda.memcpy_htod(len_d, ct.c_float(64))

grid = (1, 1)
block = (1, 1, 1)

params_ptr = getArg(8, 8, 8, 1.0, a.ctypes.data_as(c_float_p), \
64, b.ctypes.data_as(c_float_p), 64, 0.0, c.ctypes.data_as(c_float_p), 64)
#result = getConf(params_ptr)
f.prepared_call(grid, block, params_ptr, shared_size=0)

c_prod = np.empty_like(a)
cuda.memcpy_dtoh(c_prod, c_d)
#print(a*b)
print(c_prod)