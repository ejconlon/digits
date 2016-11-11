$script = <<SCRIPT
sudo apt-get update
sudo apt-get upgrade -y
cd /digits
make deps-jessie
sudo apt-get autoremove -y
SCRIPT

Vagrant.configure("2") do |config|
  config.vm.box = "debian/jessie64"
  config.vm.hostname = "digits.jessie"
  config.vm.network :private_network, ip: "192.168.0.42"
  config.vm.synced_folder ".", "/digits"
  config.vm.provision "shell", inline: $script
end
