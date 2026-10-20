#!/usr/bin/env bash
# Builds all examples from the MDNN_examples repository against THIS version of the library.
#
# The examples in "more examples/" are program.txt files, not projects; until now the only way to
# check them was to copy them into a project by hand, and three library bugs were found only by
# running the examples manually. This script wraps each program.txt in a temporary console
# project that references the library and builds it; it also builds the main MDNN_example.sln.
# It does not run them (some need data that is not in the repo); it checks they match the API.
#
# Usage:  tools/build_examples.sh <path to MDNN_examples>
set -euo pipefail

examples="$(cd "${1:?Usage: tools/build_examples.sh <path to MDNN_examples>}" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
lib="$root/MDNN/MDNN/MDNN.csproj"
plotting="$root/MDNN/MDNN.Plotting/MDNN.Plotting.csproj"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failed=0

echo "== MDNN_example.sln"
if ! dotnet build "$examples/MDNN_example/MDNN_example.sln" -c Release -nologo -v quiet; then
    echo "::error::The main example MDNN_example.sln does not build"
    failed=1
fi

i=0
while IFS= read -r -d '' program; do
    i=$((i + 1))
    name="$(basename "$(dirname "$program")")"
    dir="$work/example_$i"
    mkdir -p "$dir"
    cp "$program" "$dir/Program.cs"

    # MDNN.Plotting only for examples that actually use the chart. If every example got it,
    # ScottPlot would come along too, and an example that accidentally relies on ScottPlot
    # (which a user without MDNN.Plotting will not have) would pass here.
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
        echo "::error::Example \"$name\" does not build"
        failed=1
    fi
done < <(find "$examples" -name program.txt -not -path '*/bin/*' -not -path '*/obj/*' -print0 | sort -z)

echo "Built: main project + $i program.txt"
exit "$failed"
