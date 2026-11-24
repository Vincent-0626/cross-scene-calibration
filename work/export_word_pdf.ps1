param(
  [Parameter(Mandatory=$true)][string]$InputPath,
  [Parameter(Mandatory=$true)][string]$PdfPath,
  [switch]$UpdateAndSave
)
$ErrorActionPreference = 'Stop'
$officeApp = $null
$officeDoc = $null
try {
  $officeApp = New-Object -ComObject Word.Application
  $officeApp.Visible = $false
  $officeApp.DisplayAlerts = 0
  $officeDoc = $officeApp.Documents.Open($InputPath, $false, (-not $UpdateAndSave))
  if ($UpdateAndSave) {
    $officeDoc.Fields.Update() | Out-Null
    foreach ($tocItem in $officeDoc.TablesOfContents) { $tocItem.Update() }
    $officeDoc.Repaginate()
    $officeDoc.Save()
  }
  $officeDoc.ExportAsFixedFormat($PdfPath, 17)
  Get-Item -LiteralPath $PdfPath | Select-Object FullName,Length
}
finally {
  if ($null -ne $officeDoc) { try { $officeDoc.Close(0) } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($officeDoc) | Out-Null }
  if ($null -ne $officeApp) { try { $officeApp.Quit() } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($officeApp) | Out-Null }
}
