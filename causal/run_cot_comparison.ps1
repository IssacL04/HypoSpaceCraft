# CoT vs Standard Prompting 对比实验脚本 - PowerShell 版本
# Multi-Dataset Experiment (Node 3, 4)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "CoT vs Standard Prompting Comparison" -ForegroundColor Cyan
Write-Host "Multi-Dataset Experiment (Node 3, 4)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$CONFIG = "config/config.yaml"
$N_SAMPLES = 30
$SEED = 33550336
$QUERY_MULTIPLIER = 1.0
$RESULTS_DIR = "results/cot_comparison"

$DATASETS = @(
    "datasets/n3_observations.json",
    "datasets/n4_observations.json"
)
$NODE_NAMES = @("n3", "n4")

New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Config file: $CONFIG"
Write-Host "  Samples per dataset: $N_SAMPLES"
Write-Host "  Random seed: $SEED"
Write-Host "  Query multiplier: $QUERY_MULTIPLIER"
Write-Host "  Results directory: $RESULTS_DIR"
Write-Host ""

$TOTAL_STEPS = $DATASETS.Count * 2
$CURRENT_STEP = 0

for ($i = 0; $i -lt $DATASETS.Count; $i++) {
    $DATASET = $DATASETS[$i]
    $NODE_NAME = $NODE_NAMES[$i]
    
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Processing dataset: $NODE_NAME" -ForegroundColor Cyan
    Write-Host "Dataset file: $DATASET" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-Path $DATASET)) {
        Write-Host "WARNING: Dataset file not found: $DATASET" -ForegroundColor Yellow
        Write-Host "Skipping $NODE_NAME..." -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    
    $CURRENT_STEP++
    Write-Host "[$CURRENT_STEP/$TOTAL_STEPS] Running STANDARD mode for $NODE_NAME..." -ForegroundColor Green
    
    $stdOutput = "$RESULTS_DIR/${NODE_NAME}_standard.json"
    
    python run_causal_benchmark.py `
        --dataset $DATASET `
        --config $CONFIG `
        --n-samples $N_SAMPLES `
        --seed $SEED `
        --output $stdOutput `
        --query-multiplier $QUERY_MULTIPLIER `
        --verbose
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to run standard mode for $NODE_NAME" -ForegroundColor Red
        Write-Host ""
        continue
    }
    
    Write-Host ""
    
    $CURRENT_STEP++
    Write-Host "[$CURRENT_STEP/$TOTAL_STEPS] Running CoT mode for $NODE_NAME..." -ForegroundColor Green
    
    $cotOutput = "$RESULTS_DIR/${NODE_NAME}_cot.json"
    
    python run_causal_benchmark.py `
        --dataset $DATASET `
        --config $CONFIG `
        --n-samples $N_SAMPLES `
        --seed $SEED `
        --use-cot `
        --output $cotOutput `
        --query-multiplier $QUERY_MULTIPLIER `
        --verbose
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to run CoT mode for $NODE_NAME" -ForegroundColor Red
        Write-Host ""
        continue
    }
    
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All Experiments Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: $RESULTS_DIR" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:" -ForegroundColor Yellow

foreach ($NODE_NAME in $NODE_NAMES) {
    $stdFile = "$RESULTS_DIR/${NODE_NAME}_standard.json"
    $cotFile = "$RESULTS_DIR/${NODE_NAME}_cot.json"
    
    if (Test-Path $stdFile) {
        Write-Host "  - ${NODE_NAME}_standard.json" -ForegroundColor Green
    }
    if (Test-Path $cotFile) {
        Write-Host "  - ${NODE_NAME}_cot.json" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "To analyze and visualize results, run:" -ForegroundColor Yellow
Write-Host "  python compare_results.py --results-dir $RESULTS_DIR" -ForegroundColor White
Write-Host ""

