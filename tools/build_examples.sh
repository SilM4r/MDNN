#!/usr/bin/env bash
# Přeloží všechny příklady z repozitáře MDNN_examples proti TÉHLE verzi knihovny.
#
# Příklady v "more examples/" jsou program.txt, ne projekty — do teď je šlo ověřit jen tak,
# že si je někdo ručně zkopíroval do projektu. Tři chyby v knihovně se tak našly až
# ručním spuštěním příkladů. Tenhle skript každý program.txt zabalí do dočasného
# konzolového projektu s odkazem na knihovnu a přeloží ho; přeloží i hlavní MDNN_example.sln.
# Nespouští je (některé potřebují data, která v repu nejsou) — hlídá, že sedí na API.
#
# Použití:  tools/build_examples.sh <cesta k MDNN_examples>
set -euo pipefail

examples="$(cd "${1:?Použití: tools/build_examples.sh <cesta k MDNN_examples>}" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
lib="$root/MDNN/MDNN/MDNN.csproj"
plotting="$root/MDNN/MDNN.Plotting/MDNN.Plotting.csproj"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failed=0

echo "== MDNN_example.sln"
if ! dotnet build "$examples/MDNN_example/MDNN_example.sln" -c Release -nologo -v quiet; then
    echo "::error::Hlavní příklad MDNN_example.sln nejde přeložit"
    failed=1
fi

i=0
while IFS= read -r -d '' program; do
    i=$((i + 1))
    name="$(basename "$(dirname "$program")")"
    dir="$work/example_$i"
    mkdir -p "$dir"
    cp "$program" "$dir/Program.cs"

    # MDNN.Plotting jen příkladům, které graf opravdu používají. Kdyby ho dostal každý,
    # přišel by s ním i ScottPlot a příklad, který na ScottPlot omylem spoléhá (bez
    # MDNN.Plotting ho uživatel mít nebude), by tady prošel.
    plotting_ref=""
    if grep -qE 'UseLossChart|GraphPlotter' "$program"; then
        plotting_ref="<ProjectReference Include=\"$plotting\" />"
    fi

    cat > "$dir/Example.csproj" <<CSPROJ
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="$lib" />
    $plotting_ref
  </ItemGroup>
</Project>
CSPROJ

    echo "== $name"
    if ! dotnet build "$dir/Example.csproj" -c Release -nologo -v quiet; then
        echo "::error::Příklad „$name\" nejde přeložit"
        failed=1
    fi
done < <(find "$examples" -name program.txt -not -path '*/bin/*' -not -path '*/obj/*' -print0 | sort -z)

echo "Přeloženo: hlavní projekt + $i program.txt"
exit "$failed"
