provider "google" {
  project = "terraform-lab-sriks"   # replace with your actual project ID
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_instance" "demo_vm" {
  name         = "terraform-demo-vm"
  machine_type = "f1-micro"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
  }
}
