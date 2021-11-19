for ((argpos=1; argpos<$#; argpos++)); do
    if [ "${!argpos}" == "--image_name" ]; then
        argpos_plus1=$((argpos+1))
        image_name=${!argpos_plus1}
fi
done

echo "Image_name: " $image_name

docker build -t $image_name -f docker/Dockerfile --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
