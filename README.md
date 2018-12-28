# cuda_unified_memory_example

This repository contains code from [Unified Memory for CUDA Beginners](https://devblogs.nvidia.com/unified-memory-cuda-beginners/), and I test on [Tesla V100](https://www.nvidia.com/en-sg/data-center/tesla-v100/).  

(1) Compile and profile `add_grid.cu`:  

	$ nvcc add_grid.cu -o add_grid
	$ nvprof ./add_grid
	==83427== NVPROF is profiling process 83427, command: ./add_grid
	Max error: 0
	==83427== Profiling application: ./add_grid
	==83427== Profiling result:
	            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
	 GPU activities:  100.00%  5.4755ms         1  5.4755ms  5.4755ms  5.4755ms  add(int, float*, float*)
	      API calls:   93.44%  359.91ms         2  179.96ms  97.986us  359.81ms  cudaMallocManaged
	                    3.12%  12.012ms       384  31.281us  1.2680us  6.5458ms  cuDeviceGetAttribute
	                    1.59%  6.1155ms         4  1.5289ms  1.2442ms  2.2906ms  cuDeviceTotalMem
	                    1.42%  5.4802ms         1  5.4802ms  5.4802ms  5.4802ms  cudaDeviceSynchronize
	                    0.25%  973.81us         2  486.91us  474.33us  499.48us  cudaFree
	                    0.11%  440.19us         4  110.05us  83.362us  157.54us  cuDeviceGetName
	                    0.05%  207.22us         1  207.22us  207.22us  207.22us  cudaLaunchKernel
	                    0.00%  13.797us         8  1.7240us  1.3020us  2.9550us  cuDeviceGet
	                    0.00%  13.325us         4  3.3310us  1.4120us  5.3510us  cuDeviceGetPCIBusId
	                    0.00%  6.7590us         3  2.2530us  1.3130us  3.6500us  cuDeviceGetCount
	                    0.00%  6.6140us         4  1.6530us  1.4440us  1.8420us  cuDeviceGetUuid
	
	==83427== Unified Memory profiling result:
	Device "Tesla V100-PCIE-16GB (0)"
	   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
	     173  47.353KB  4.0000KB  980.00KB  8.000000MB  1.069152ms  Host To Device
	      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  342.0800us  Device To Host
	       8         -         -         -           -  5.461280ms  Gpu page fault groups
	Total CPU Page faults: 36  

(2) Compile and profile `add_grid_init.cu`:    

	$ nvcc add_grid_init.cu -o add_grid_init
	$ nvprof ./add_grid_init
	==87851== NVPROF is profiling process 87851, command: ./add_grid_init
	Max error: 0
	==87851== Profiling application: ./add_grid_init
	==87851== Profiling result:
	            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
	 GPU activities:   99.67%  4.9634ms         1  4.9634ms  4.9634ms  4.9634ms  init(int, float*, float*)
	                    0.33%  16.192us         1  16.192us  16.192us  16.192us  add(int, float*, float*)
	      API calls:   95.54%  326.68ms         2  163.34ms  95.892us  326.58ms  cudaMallocManaged
	                    1.46%  4.9917ms         2  2.4958ms  20.961us  4.9707ms  cudaDeviceSynchronize
	                    1.35%  4.6146ms         4  1.1537ms  853.25us  1.2638ms  cuDeviceTotalMem
	                    1.24%  4.2258ms       384  11.004us     875ns  462.08us  cuDeviceGetAttribute
	                    0.19%  645.95us         2  322.97us  293.89us  352.06us  cudaFree
	                    0.11%  382.80us         2  191.40us  179.71us  203.09us  cudaLaunchKernel
	                    0.10%  354.08us         4  88.518us  83.147us  97.597us  cuDeviceGetName
	                    0.00%  14.375us         4  3.5930us  1.5970us  6.0770us  cuDeviceGetPCIBusId
	                    0.00%  12.702us         8  1.5870us  1.2930us  2.4770us  cuDeviceGet
	                    0.00%  6.3830us         3  2.1270us  1.3270us  3.2840us  cuDeviceGetCount
	                    0.00%  5.2920us         4  1.3230us     911ns  1.6210us  cuDeviceGetUuid
	
	==87851== Unified Memory profiling result:
	Device "Tesla V100-PCIE-16GB (0)"
	   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
	     189  43.344KB  4.0000KB  860.00KB  8.000000MB  1.097184ms  Host To Device
	      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  342.5280us  Device To Host
	      10         -         -         -           -  4.947712ms  Gpu page fault groups
	Total CPU Page faults: 36
	
(3) Compile and profile `add_grid_many.cu`:  

	$ nvcc add_grid_many.cu -o add_grid_many
	$ nvprof ./add_grid_many
	==91203== NVPROF is profiling process 91203, command: ./add_grid_many
	Max error: 99
	==91203== Profiling application: ./add_grid_many
	==91203== Profiling result:
	            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
	 GPU activities:  100.00%  6.7171ms       100  67.170us  15.296us  5.1706ms  add(int, float*, float*)
	      API calls:   94.58%  356.90ms         2  178.45ms  49.913us  356.85ms  cudaMallocManaged
	                    1.91%  7.2095ms       100  72.095us  19.292us  5.1783ms  cudaDeviceSynchronize
	                    1.24%  4.6890ms         4  1.1723ms  858.10us  1.2979ms  cuDeviceTotalMem
	                    1.13%  4.2650ms       384  11.106us     850ns  442.47us  cuDeviceGetAttribute
	                    0.85%  3.2044ms       100  32.044us  24.555us  164.36us  cudaLaunchKernel
	                    0.18%  691.24us         2  345.62us  322.07us  369.18us  cudaFree
	                    0.10%  363.61us         4  90.903us  84.461us  97.819us  cuDeviceGetName
	                    0.00%  13.644us         8  1.7050us  1.2830us  2.5700us  cuDeviceGet
	                    0.00%  12.049us         4  3.0120us  1.6830us  4.9230us  cuDeviceGetPCIBusId
	                    0.00%  6.0940us         3  2.0310us  1.2760us  3.0290us  cuDeviceGetCount
	                    0.00%  5.3850us         4  1.3460us     913ns  1.6900us  cuDeviceGetUuid
	
	==91203== Unified Memory profiling result:
	Device "Tesla V100-PCIE-16GB (0)"
	   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
	     165  49.648KB  4.0000KB  960.00KB  8.000000MB  1.044896ms  Host To Device
	      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  342.6880us  Device To Host
	       8         -         -         -           -  5.151744ms  Gpu page fault groups
	Total CPU Page faults: 36  

(4) Compile and profile `add_grid_prefetch.cu`:  

	$ nvcc add_grid_prefetch.cu -o add_grid_prefetch
	$ nvprof ./add_grid_prefetch
	==94013== NVPROF is profiling process 94013, command: ./add_grid_prefetch
	Max error: 0
	==94013== Profiling application: ./add_grid_prefetch
	==94013== Profiling result:
	            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
	 GPU activities:  100.00%  16.544us         1  16.544us  16.544us  16.544us  add(int, float*, float*)
	      API calls:   95.00%  299.95ms         2  149.97ms  78.437us  299.87ms  cudaMallocManaged
	                    2.08%  6.5619ms       384  17.088us     969ns  681.51us  cuDeviceGetAttribute
	                    1.59%  5.0223ms         4  1.2556ms  1.1375ms  1.5577ms  cuDeviceTotalMem
	                    0.50%  1.5884ms         2  794.20us  198.38us  1.3900ms  cudaMemPrefetchAsync
	                    0.37%  1.1737ms         1  1.1737ms  1.1737ms  1.1737ms  cudaDeviceSynchronize
	                    0.25%  784.40us         2  392.20us  328.10us  456.30us  cudaFree
	                    0.15%  480.27us         4  120.07us  111.98us  133.31us  cuDeviceGetName
	                    0.05%  152.11us         1  152.11us  152.11us  152.11us  cudaLaunchKernel
	                    0.00%  10.944us         4  2.7360us  2.0270us  3.2220us  cuDeviceGetPCIBusId
	                    0.00%  10.067us         8  1.2580us  1.0140us  1.8720us  cuDeviceGet
	                    0.00%  7.7880us         1  7.7880us  7.7880us  7.7880us  cudaGetDevice
	                    0.00%  5.2150us         3  1.7380us     997ns  2.6820us  cuDeviceGetCount
	                    0.00%  4.5870us         4  1.1460us  1.0900us  1.2410us  cuDeviceGetUuid
	
	==94013== Unified Memory profiling result:
	Device "Tesla V100-PCIE-16GB (0)"
	   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
	       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  697.1520us  Host To Device
	      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  342.3360us  Device To Host
	Total CPU Page faults: 36
	
