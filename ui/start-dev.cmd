@echo off

REM Install dependencies if not exist
if not exist node_modules (
    echo Installing dependencies...
    npm install
)

REM Load environment variables
for /f "tokens=*" %%a in ('type ..\.env ^| findstr /v "^#"') do set %%a

REM Run the application with nodemon
echo Starting the application...
nodemon --watch pages --watch components --exec "next dev"