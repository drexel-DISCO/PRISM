# PRISM (Performance-oriented real-time scheduler for machine learning operations on a heterogeneous NSoC)


This is a Python implementation of our real-time scheduler. A sample script is provided (performance.sh) to execute the scheduler. The script generates the average execution time per input batch. Throughput can be obtained by computing the inverse of this execution time. There are two types of execution time estimaates provided via the script. First, the script computes the throughput for the given batch size as configured in the parameters.py file. Second, the script computes the long-term throughput by iterating through different batch sizes. Currently, it uses batch size of 2, 4, 8, 16, 32, & 64.

The script can be configured to model an NSoC platform. This can be performed via the parameters.py file. This file shows the configuration and supported operations for the GrAI hardware. Execution time for each operation of a machine learning model is provided in the extime folder. This is created as dictionary with two timing parameters: one for the NPU and the other for the CPU.

The script suppports three schedulers.

1) random: Here, operations are randomly mapped to either a CPU or an NPU.

2) npufirst: This is an NPU-First policy.  It uses a GPP to schedule only the operations not supported on the NPU.

3) prism: This is our PRISM scheduler

The ordering is performed using self-timed execution.

### Schedule extraction
All sample scripts are provided for xception moddel.

Results are saved in the results folder "results/performance/prism/grai/NPU_<n_npu>.CPU_<n_cpu>/schedules/".

The output is a pickle file containing the schedule, which is implemented as a Python class. The class structure is described in the file scheduler_class.py.

The following code is to read and process this pickle file.

fname = 'results/performance/prism/grai/NPU_<n_npu>.CPU_<n_cpu>/schedules/' + model_name + '.pkl'

with open(fname, "rb") as sdata:

    sch = pickle.load(sdata)

s = sch[0]

print(s.start_times())      #prints the execution start times of all operations of the model. The length of this list is same as the number of operations.

print(s.end_times())        #prints the execution end times of all operations of the model.

print(s.computing_cores())  #prints the processing elements (CPU or NPU) that executes these operations.

print(s.computing_ids())    #prints the ids of the resource that executes these operations.

print(s.map())              #prints the mapping of each operation to a resource.

## Reference
If you find this code useful in your research, please cite the following paper:

Anup Das, "Real-Time Scheduling of Machine Learning Operations on Heterogeneous Neuromorphic SoC", 20th ACM/IEEE International Conference on Formal Methods and Models for System Design (MEMOCODE'22)

@inproceedings{prism,

title={Real-Time Scheduling of Machine Learning Operations on Heterogeneous Neuromorphic SoC}

author={Das, Anup},

booktitle ={International Conference on Formal Methods and Models for System Design (MEMOCODE)},

year={2022},

publisher={ACM/IEEE}

}



### Sample Output (prism mapping)
> ./performance.sh 

Enter the model: xception

Enter algorithm (chose between random, npufirst, prism): prism

Enter the architecture (chose between mubrain, speck, and grai): grai

Enter number of npus: 4

Enter number of cpus: 2

[info] Scheduling  xception  ...

[hco] Starting the Hill Climbing Optimization

[hco] Pass  0  performance =  2183219.796971777

[hco] New performance =  2178719.796971777

[hco] New performance =  2159119.796971777

[hco] New performance =  2148619.796971777

...

[hco] Pass  1  performance =  1064300.6650120076

[hco] New performance =  1061900.6650120076

[hco] New performance =  1058500.6650120076

[hco] New performance =  1053300.6650120076

[hco] New performance =  1051400.6650120076

[hco] Pass  2  performance =  1051400.6650120076

[hco] New performance =  1051300.6650120076

[hco] New performance =  1048400.6650120077

...

[hco] Pass  3  performance =  1045100.6650120077

[hco] Endiing the Hill Climbing Optimization

[info] done

xception : execution time for  16  batches =  [1045100.6650120077] us

Simulation time for graph compilation (wall clock time) =  1.2720916271209717  seconds

Simulation time for schedule generation (wall clock time) =  5543.874638080597  seconds

### Sample Output (random mapping)
> ./performance.sh 

Enter the model: xception

Enter algorithm (chose between random, npufirst, prism): random

Enter the architecture (chose between mubrain, speck, and grai): grai

Enter number of npus: 4

Enter number of cpus: 2

[info] Scheduling  xception  ...

[info] done

xception : execution time for  16  batches =  [2183219.796971777] us

Simulation time for graph compilation (wall clock time) =  1.3997037410736084  seconds

Simulation time for schedule generation (wall clock time) =  14.996781587600708  seconds

[info] Computing average throughput of  xception  ...

[info] Scheduling  2  batches

[info] Scheduling  4  batches

[info] Scheduling  8  batches

[info] Scheduling  16  batches

[info] Scheduling  32  batches

[info] Scheduling  64  batches

[info] Scheduling  128  batches

[info] done

xception : average model execution time per batch =  144489.07836541932 us

Simulation time for throughput computation (wall clock time) =  7871.3557505607605  seconds

### Sample Output (npufirst mapping)
> ./performance.sh 

Enter the model: xception

Enter algorithm (chose between random, npufirst, prism): npufirst

Enter the architecture (chose between mubrain, speck, and grai): grai

Enter number of npus: 4

Enter number of cpus: 2

[info] Scheduling  xception  ...

[info] done

xception : execution time for  16  batches =  [1064300.6650120076] us

Simulation time for graph compilation (wall clock time) =  1.2708649635314941  seconds

Simulation time for schedule generation (wall clock time) =  13.350368022918701  seconds

[info] Computing average throughput of  xception  ...

[info] Scheduling  2  batches

[info] Scheduling  4  batches

[info] Scheduling  8  batches

[info] Scheduling  16  batches

[info] Scheduling  32  batches

[info] Scheduling  64  batches

[info] Scheduling  128  batches

[info] done

xception : average model execution time per batch =  81885.69693808591 us

Simulation time for throughput computation (wall clock time) =  7439.766078710556  seconds
