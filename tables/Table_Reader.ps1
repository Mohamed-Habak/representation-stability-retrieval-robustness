$folder = $PSScriptRoot
$output = Join-Path $folder "all_tables.csv"

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false

$result = @()

Get-ChildItem "$folder\*.xlsx" | ForEach-Object {
    $file = $_
    $book = $excel.Workbooks.Open($file.FullName)

    foreach ($sheet in $book.Worksheets) {
        $result += "TABLE: $($file.BaseName) - $($sheet.Name)"

        $data = $sheet.UsedRange.Value2

        if ($data -is [array]) {
            for ($r = 1; $r -le $data.GetLength(0); $r++) {
                $row = for ($c = 1; $c -le $data.GetLength(1); $c++) {
                    $data[$r, $c]
                }
                $result += ($row -join ",")
            }
        }
        else {
            $result += [string]$data
        }

        $result += ""
    }

    $book.Close($false)
}

$excel.Quit()

$result | Set-Content $output -Encoding UTF8

Write-Host "Created: $output"