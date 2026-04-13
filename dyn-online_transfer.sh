# For testing ml dynamics only
 
set -e

MISSION=bomd-sz-online_transfer

#rm -rf $MISSION
if [ ! -d "$MISSION" ];
then
    #echo $MISSION
    mkdir $MISSION
fi
cp sample_geometries/dyn-ML/$MISSION.com $MISSION/bomd.com
cp sample_geometries/dyn-ML/$MISSION.toml $MISSION/$MISSION.toml

python run_dynamics.py NN_test_files/dyn-ML/$MISSION.toml

echo Dynamics $MISSION finished.

