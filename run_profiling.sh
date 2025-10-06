#!/bin/bash

# CUDA Particle Filter Profiling Script
# This script runs the particle filter with various profiling tools

echo "=============================================="
echo "CUDA Particle Filter Performance Analysis"
echo "=============================================="

# Check if nvprof is available
if ! command -v nvprof &> /dev/null; then
    echo "Warning: nvprof not found. Install CUDA Toolkit with profiling tools."
    exit 1
fi

# Check if Nsight is available
if command -v ncu &> /dev/null; then
    NSIGHT_AVAILABLE=1
    echo "Nsight Compute (ncu) found - will generate detailed reports"
else
    NSIGHT_AVAILABLE=0
    echo "Nsight Compute (ncu) not found - using nvprof only"
fi

# Build the project
echo "Building project..."
make clean && make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "=============================================="
echo "1. Running with nvprof (basic profiling)"
echo "=============================================="

# Run with nvprof for basic kernel timing
nvprof --print-gpu-trace ./particle_filter_main 2> nvprof_trace.log

echo "=============================================="
echo "2. Running with detailed nvprof metrics"
echo "=============================================="

# Run with detailed metrics
nvprof --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,gld_efficiency,gst_efficiency \
       --print-gpu-summary ./particle_filter_main 2> nvprof_metrics.log

echo "=============================================="
echo "3. Running with Nsight Compute (if available)"
echo "=============================================="

if [ $NSIGHT_AVAILABLE -eq 1 ]; then
    # Run with Nsight Compute for detailed analysis
    ncu --target-processes all --metrics sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
         --print-summary per-kernel ./particle_filter_main 2> ncu_report.log
fi

echo "=============================================="
echo "4. Generating performance report"
echo "=============================================="

# Extract key metrics from logs
echo "Performance Analysis Report" > performance_report.txt
echo "==========================" >> performance_report.txt
echo "" >> performance_report.txt

# Extract kernel times from nvprof
echo "Kernel Execution Times:" >> performance_report.txt
grep "predict_kernel\|update_weights_kernel\|normalize_weights_kernel\|resample_optimized_kernel\|set_uniform_weights_kernel" nvprof_trace.log | \
awk '{print $9 " " $10 " " $11}' | sort -k3 -n >> performance_report.txt

echo "" >> performance_report.txt
echo "Key Metrics from nvprof:" >> performance_report.txt
grep -A 10 "GPU summary" nvprof_metrics.log >> performance_report.txt

echo "" >> performance_report.txt
echo "Recommendations:" >> performance_report.txt
echo "- Check if achieved occupancy is > 50%" >> performance_report.txt
echo "- Look for memory efficiency < 80% (potential optimization)" >> performance_report.txt
echo "- Warp execution efficiency should be > 90%" >> performance_report.txt
echo "- Profile with different particle counts for scalability" >> performance_report.txt

echo "=============================================="
echo "Profiling complete!"
echo "Check the following files:"
echo "- nvprof_trace.log: Detailed kernel traces"
echo "- nvprof_metrics.log: GPU metrics and summary"
if [ $NSIGHT_AVAILABLE -eq 1 ]; then
    echo "- ncu_report.log: Nsight Compute detailed analysis"
fi
echo "- performance_report.txt: Summary and recommendations"
echo "=============================================="