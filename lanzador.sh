#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'

echo -e "----- ${RED}Lanzador del mapeador${NC} -----"

cmake CMakeLists.txt

echo "Compilando la aplicación..."

make

reset
echo "Introduce el nombre del vídeo a procesar"
echo ""
echo -e "${RED}ADVERTENCIA${NC}: Es importante que el vídeo se encuentre en la carpeta: vidsDrone"
echo ""
read;
videoName="./vidsDrone/${REPLY}"
reset
echo "Vídeo seleccionado: $videoName"
echo ""
echo ""
SalirBucle=0
while [ ${SalirBucle} -eq 0 ]
do
    echo -e "${BLUE}1)${NC} Ejecucion rapida        (mas imperfecciones)"
    echo -e "${PURPLE}2)${NC} Ejecucion lenta         (menos imperfecciones pero más tiempo de ejecucion)"
    echo -e "${CYAN}3)${NC} Ejecucion personalizada (se pedirán argumentos para la ejecucion)"
    echo ""
    read
    opcion="${REPLY}"
    if [ ${REPLY} -eq 1 ]
    then
        SalirBucle=1
    fi
    if [ ${REPLY} -eq 2 ]
    then
        SalirBucle=1
    fi
    if [ ${REPLY} -eq 3 ]
    then
        SalirBucle=1
    fi
    reset
done
reset

# Argumentos:
# --video <videoname>
# --segundos <int>
# --match_conf <float>
# --compose_megapix <float> --> SOLO USADO PARA EJECUCION RAPIDA
# --use_composer --> Si se usa se refinan los bordes
# --output <result_img>

echo "Introduce el nombre que tendrá la imagen resultante (sin el formato)"
echo ""
read;
resultName=${REPLY}
reset
echo "Nombre del plano resultado: $resultName.jpg"

if [ ${opcion} -eq 1 ]
then
    echo "Realizando ejecución rápida..."
    ./mapeadoConDrones --video $videoName --segundos 15 --match_conf 0.55 --compose_megapix 0.6 --use_composer --output $resultName

fi
if [ ${opcion} -eq 2 ]
then
    echo "Realizando ejecución lenta..."
    ./mapeadoConDrones --video $videoName --segundos 22 --match_conf 0.5 --use_composer --output $resultName
fi
if [ ${opcion} -eq 3 ]
then
    echo "Modificación de parámetros: "
    echo ""
    echo "1) Número de segundos del vídeo a procesar: "
    read;
    segundos=${REPLY}
    echo "2) Confianza para selección de frames clave [0.0-1.0]: "
    read;
    confianza=${REPLY}

    usarRefinador=0
    while (( usarRefinador != 1 && usarRefinador != 2 ))
    do
        echo "3) ¿Usar refinador de bordes?"
        echo "   1-> SÍ"
        echo "   2-> NO"
        read;
        usarRefinador=${REPLY}
    done

    reset
    echo "Realizando ejecución personalizada..."
    if [ $usarRefinador -eq 1 ]
    then
        ./mapeadoConDrones --video $videoName --segundos $segundos --match_conf $confianza --use_composer --output $resultName
    else
        ./mapeadoConDrones --video $videoName --segundos $segundos --match_conf $confianza --output $resultName
    fi

fi
