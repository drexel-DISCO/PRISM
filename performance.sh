#!/bin/bash

read -p "Enter the model: " MODEL
read -p "Enter algorithm (chose between random, npufirst, prism): " ALGO
read -p "Enter the architecture (chose between mubrain, speck, and grai): " ARCH
read -p "Enter number of npus: " NPUS
read -p "Enter number of cpus: " CPUS

RND="random"
NPUFIRST="npufirst"
PRISM="prism"

MUBRAIN="mubrain"
SPECK="speck"
GRAI="grai"

#configure the architecture here
if [ "$ARCH" = "$MUBRAIN" ]; then
    sed -i '/NPU_SUPPORTED_OPERATIONS/c\NPU_SUPPORTED_OPERATIONS = MUBRAIN_SUPPORTED_OPERATIONS' parameters.py
elif [ "$ARCH" = "$SPECK" ]; then
    sed -i '/NPU_SUPPORTED_OPERATIONS/c\NPU_SUPPORTED_OPERATIONS = SPECK_SUPPORTED_OPERATIONS' parameters.py
elif [ "$ARCH" = "$GRAI" ]; then
    sed -i '/NPU_SUPPORTED_OPERATIONS/c\NPU_SUPPORTED_OPERATIONS = GRAI_SUPPORTED_OPERATIONS' parameters.py
else
    echo "$ARCH is not currently supported"
    exit 0
fi

#configure the npus and cpus
sed -i "/n_cpus/c\n_cpus = $CPUS" parameters.py
sed -i "/n_npus/c\n_npus = $NPUS" parameters.py

#configure the compiler
if [ "$ALGO" = "$RND" ]; then
    SCHD="rnd"
elif [ "$ALGO" = "$NPUFIRST" ]; then
    SCHD="npufirst"
elif [ "$ALGO" = "$PRISM" ]; then
    SCHD="prism"
else
    echo "$ALGO is not currently supported"
    exit 0
fi

OUTDIR="results/performance/$ALGO/$ARCH/NPU_$NPUS.CPU_$CPUS"
mkdir -p "$OUTDIR/schedules" "$OUTDIR/logs"

#run the compiler
python3 compiler.py -model $MODEL -schd $SCHD -out "$OUTDIR/schedules"
