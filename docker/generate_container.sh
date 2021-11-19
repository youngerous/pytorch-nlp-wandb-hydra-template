for ((argpos=1; argpos<$#; argpos++)); do
    if [ "${!argpos}" == "--container_name" ]; then
        argpos_plus1=$((argpos+1))
        container_name=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--image_name" ]; then
        argpos_plus1=$((argpos+1))
        image_name=${!argpos_plus1}
    fi
done

echo "Container_name: " $container_name
echo "Image_name: " $image_name

# --gpus '"device=0,1"'
docker run --gpus '"device=all"' -td --ipc=host --name $container_name\
	-v ~/repo:/repo\
	-v /etc/passwd:/etc/passwd\
    -v /etc/localtime:/etc/localtime:ro\
	-e TZ=Asia/Seoul\
	-p 8888:8888 -p 6006:6006 $image_name
