# FaSNet-TAC-pyTorch
Full implementation of "End-to-end microphone permutation and number invariant multi-channel speech separation" (Interspeech 2020)



## Plan

- [x] Data pre-processing
- [x] Training
- [x] Inference
- [x] Separate


You can use our code by changing data_script/tr.scp, cv.scp, tt.scp as your data directory.

```bash
# In scp file

D:/MC_Libri_fixed/tr # your path
20000 # the number of samples
```

Data generation script: https://github.com/yluo42/TAC/tree/master/data

## Reference
https://github.com/yluo42/TAC/
