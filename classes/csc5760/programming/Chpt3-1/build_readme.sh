#! /bin/bash

set -x

PANDOC=/usr/bin/pandoc

$PANDOC --standalone -V geometry:margin=1in -o readme.pdf readme.md


# build the zip file
rm -rf Chpt3-1_T00215814
rm -f Chpt3-1_T00215814.zip
mkdir -p Chpt3-1_T00215814
cp -r ./src ./Chpt3-1_T00215814
cp -r ./include ./Chpt3-1_T00215814
cp CMakeLists.txt Chpt3-1_T00215814
cp readme.md Chpt3-1_T00215814
cp readme.pdf Chpt3-1_T00215814
zip -r Chpt3-1_T00215814.zip Chpt3-1_T00215814
rm -rf Chpt3-1_T00215814
