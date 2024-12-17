# stop and remove the images
# docker stop shopping_admin forum gitlab shopping
# docker remove shopping_admin forum gitlab shopping


docker stop shopping_admin forum shopping
docker remove shopping_admin forum shopping
# start the images
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
# docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
