$script = <<SCRIPT
sudo apt-get update
cd /digits
make deps-linux
sudo apt-get autoremove -y
SCRIPT

Vagrant.configure("2") do |config|
  #config.vm.box = "debian/contrib-jessie64"
  #config.vm.hostname = "digits.jessie"
  config.vm.box = "ubuntu/trusty64"
  config.vm.hostname = "digits.trusty"
  config.vm.network :private_network, ip: "192.168.0.42"
  config.vm.synced_folder ".", "/digits", type: "virtualbox"
  config.vm.provision "shell", inline: $script
  config.vm.provider "virtualbox" do |v|
    v.memory = 2048
  end
end
