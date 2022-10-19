#!/bin/bash

#arg 1 system name
#arg 2 no of parameters
#arg 3 starting interval
#arg 4 last interval 
#arg 5 acts filename

sys=$1
params=$2
l=$3
u=$4  
acts=$5

echo '[System]' > $acts
echo 'Name: '$sys >> $acts
echo '[Parameter]' >> $acts

p=1
while [ $p -le $params ]
do
	echo -n "p"$p "(int) : " >> $acts
	interval=$l
	while [ $interval -le $u ]
	do
		echo -n $interval >> $acts
		if [ $interval != $u ] 	
		then
			echo -n "," >> $acts
		fi
		interval=`expr $interval + 1`
	done
	echo >> $acts
	p=`expr $p + 1`
done  




