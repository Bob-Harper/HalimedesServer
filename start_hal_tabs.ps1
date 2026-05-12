$wt = "$env:LOCALAPPDATA\Microsoft\WindowsApps\wt.exe"

$gateway = "C:\Halimedes\venv\Scripts\python.exe C:\Halimedes\gateway\hal_server_gateway.py"
$vosk    = "C:\Halimedes\venv\Scripts\python.exe C:\Halimedes\servers\vosk_server.py"
$llm     = "C:\Halimedes\venv\Scripts\python.exe C:\Halimedes\servers\llm_server.py"
$ssh     = "ssh msutt@192.168.0.102"

& $wt -p "Windows PowerShell" -d "C:\Halimedes" --tabColor "#DCDCDC" --title "Gateway" powershell -NoExit -Command $gateway `; `
new-tab -p "Windows PowerShell" -d "C:\Halimedes" --tabColor "#C0C0C0" --title "Vosk" powershell -NoExit -Command $vosk `; `
new-tab -p "Windows PowerShell" -d "C:\Halimedes" --tabColor "#FFD700" --title "LLM" powershell -NoExit -Command $llm `; `
new-tab --tabColor "#F8F8FF" --title "Halimedes" powershell -NoExit -Command $ssh
