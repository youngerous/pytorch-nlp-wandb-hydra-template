for ((argpos=1; argpos<$#; argpos++)); do
    if [ "${!argpos}" == "--container_name" ]; then
        argpos_plus1=$((argpos+1))
        container_name=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--image_name" ]; then
        argpos_plus1=$((argpos+1))
        image_name=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--port_jupyter" ]; then
        argpos_plus1=$((argpos+1))
        port_jupyter=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--port_tensorboard" ]; then
        argpos_plus1=$((argpos+1))
        port_tensorboard=${!argpos_plus1}
    fi
done

echo "Container Name: " $container_name
echo "Image Name: " $image_name
echo "Jupyter Port #: " $port_jupyter
echo "Tensorboard Port #: " $port_tensorboard

# --gpus '"device=0,1"'
docker run --gpus '"device=all"' -td --ipc=host --name $container_name\
	-v ~/repo:/repo\
	-v /etc/passwd:/etc/passwd\
    -v /etc/localtime:/etc/localtime:ro\
	-e TZ=Asia/Seoul\
	-p $port_jupyter:$port_jupyter -p $port_tensorboard:$port_tensorboard $image_name
