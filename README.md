# OneAPI Crystal

This library is the DPC++ porting (and extension) of the Crystal library, originally written in CUDA.

It Implements a collection of block-wide device functions that can be used to implement high performance implementations of SQL queries on CPUs and GPUs.

## Usage on Intel® DevCloud

Above you can find a few details on how to run this code on Intel CPUs and GPUs using Intel DevCloud platform, assuming you cloned the repository already

### Intel® Xeon® Gold 6128 CPUs
```shell
>> cd oneapi_crystal
>> qsub -I -d .
```
Now you have an interactive shell on a CPU node.

### Intel® UHD Graphics P630 (iGPU):

```shell
>> cd oneapi_crystal
>> qsub -I -l nodes=2:gpu:ppn=2 -d .
```
Now you have an interactive shell on a (integrated) GPU node.

### Intel® Iris® Xe MAX Graphics (dGPU):

```shell
>> cd oneapi_crystal
>> qsub -I -l nodes=1:iris_xe_max:ppn=2 -d .
```
Now you have an interactive shell on a (discrete) GPU node.

## Run the queries

To run the star schema benchmark implementation:

```bash
# Generate the test generator / transformer binaries
cd ssb/dbgen
make
cd ../loader
make 
cd ../../

# Generate the test data and transform into columnar layout
# Substitute <SF> with appropriate scale factor (eg: 1)
python util.py ssb <SF> gen
python util.py ssb <SF> transform
```

configure the benchmark settings

```bash
vi queries/ssb_utils.h
# edit BASE_PATH in ssb_utils.h with the location
# of oneapi_crystal 
```

now build the queries

```bash
make queries
```

and run, say q11

```
./build/q11
```

## Run the operators

Compile the operators running 
```shell
make operators
```
then benchmark, say the hash join like this

```shell
./build/join <table_size>
```
