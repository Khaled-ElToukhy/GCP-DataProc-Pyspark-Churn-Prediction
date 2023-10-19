# Configure the GCP provider
provider "google" {
  credentials = file("D:/DE Projects/Project_2/careful-triumph-401501-6b50ffcc9dd3.json")
  project     = "careful-triumph-401501"
  region      = "us-central1" # Replace with the desired region
}

# Create a Cloud Storage bucket
resource "google_storage_bucket" "my_bucket" {
  name     = "my-storage-bucket"
  location = "europe-west2" # Replace with the desired region
}

# Create a Dataproc cluster
resource "google_dataproc_cluster" "my_cluster" {
  name           = "my-dataproc-cluster"
  project        = "careful-triumph-401501"
  region         = "us-central1" # Replace with the desired region
  cluster_config {
    master_config {
      num_instances = 1
      machine_type  = "n1-standard-2" # Replace with the desired machine type
    }
    worker_config {
      num_instances = 2
      machine_type  = "n1-standard-2" # Replace with the desired machine type
    }
    software_config {
      image_version = "1.5" # Replace with the desired Dataproc version
      optional_components = ["JUPYTER", "DOCKER"] # Add any additional components you want
    }
  }
}
