# Engineered by uncoalesced
# Launch gate_step12000_rest_and_eval.py sentinel in background

$python = "E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe"
$script = "E:\Parentheses\parentheses-0.9\scripts\gate_step12000_rest_and_eval.py"
$stdout = "E:\Parentheses\parentheses-0.9\out\gate_step12000.log"
$stderr = "E:\Parentheses\parentheses-0.9\out\gate_step12000_err.log"

$p = Start-Process -FilePath $python -ArgumentList "-u $script" -WorkingDirectory "E:\Parentheses\parentheses-0.9" -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
Write-Output "Sentinel PID: $($p.Id)"
Get-Content $stdout -ErrorAction SilentlyContinue
