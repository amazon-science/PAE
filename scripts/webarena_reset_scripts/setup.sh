docker start gitlab
docker start shopping
docker start shopping_admin
docker start forum
docker start kiwix33
cd /home/ubuntu/openstreetmap-website/
docker compose start

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://ec2-18-246-217-25.us-west-2.compute.amazonaws.com:7770" # no trailing /
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://ec2-18-246-217-25.us-west-2.compute.amazonaws.com:7770/" WHERE path = "web/secure/base_url";'
# remove the requirement to reset password
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://ec2-18-246-217-25.us-west-2.compute.amazonaws.com:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://ec2-18-246-217-25.us-west-2.compute.amazonaws.com:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

# docker exec gitlab sed -i "s|^external_url.*|external_url 'http://ec2-18-246-217-25.us-west-2.compute.amazonaws.com:8023'|" /etc/gitlab/gitlab.rb
# docker exec gitlab gitlab-ctl reconfigure
