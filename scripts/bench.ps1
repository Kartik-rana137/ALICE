# Speed benchmark for ALICE candidate models using llama-bench.
# Usage:  .\scripts\bench.ps1 -Model Qwen3.5-4B-Q4_K_M [-Ngl 99] [-NCpuMoe 0] [-Label "note"]
# Appends one CSV row per test to benchmarks/results/speed.csv (tracked in git).
param(
  [Parameter(Mandatory)][string]$Model,
  [int]$Ngl = 99,
  [int]$NCpuMoe = 0,      # MoE: number of layers whose expert weights stay on CPU
  [int]$PromptTokens = 512,
  [int]$GenTokens = 128,
  [int]$Reps = 3,
  [string]$Label = ""
)
$root = Split-Path $PSScriptRoot -Parent
$bench = Join-Path $root "tools\llama.cpp\llama-bench.exe"
$gguf  = Join-Path $root "models\$Model.gguf"
$out   = Join-Path $root "benchmarks\results\speed.csv"
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

$args = @("-m", $gguf, "-ngl", $Ngl, "-p", $PromptTokens, "-n", $GenTokens, "-r", $Reps, "-o", "json")
if ($NCpuMoe -gt 0) { $args += @("--n-cpu-moe", $NCpuMoe) }

$json = & $bench @args 2>$null | Out-String
$rows = $json | ConvertFrom-Json
$date = Get-Date -Format "yyyy-MM-dd"
if (-not (Test-Path $out)) {
  "date,model,ngl,n_cpu_moe,test,tokens_per_s,stddev,label" | Out-File $out -Encoding utf8
}
foreach ($r in $rows) {
  $test = if ($r.n_gen -gt 0) { "tg$($r.n_gen)" } else { "pp$($r.n_prompt)" }
  "$date,$Model,$Ngl,$NCpuMoe,$test,$([math]::Round($r.avg_ts,2)),$([math]::Round($r.stddev_ts,2)),$Label" |
    Out-File $out -Append -Encoding utf8
  "{0,-34} ngl={1,-3} {2,-7} {3,8:N1} t/s" -f $Model, $Ngl, $test, $r.avg_ts
}
