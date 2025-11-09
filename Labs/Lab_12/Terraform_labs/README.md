# Terraform Beginner Lab - Google Cloud Platform (GCP)

## Overview

This lab provides a step-by-step guide to learning the basics of Terraform using Google Cloud Platform (GCP). Students will learn how to install, configure, and use Terraform to create, manage, and destroy cloud infrastructure resources in a reproducible way.

By completing this lab, you will:

- Understand Terraform's purpose and workflow
- Configure authentication with GCP
- Write and execute a basic Terraform configuration
- Manage and destroy resources safely

---

## Prerequisites

1. Google Cloud Platform account with billing enabled
2. Google Cloud SDK (gcloud CLI) installed and configured
3. Terraform CLI installed
4. A GCP project with a billing account linked

---

## Step 1: Install and Configure the Google Cloud SDK

### macOS / Linux

Open a terminal and install the SDK:

```bash
curl https://sdk.cloud.google.com | bash
```

Restart your terminal, then initialize the SDK:

```bash
gcloud init
```

Follow the prompts to log in, create or select a project, and set a default region and zone.

Verify the installation:

```bash
gcloud --version
```

---

## Step 2: Create a Service Account and Key

In the GCP Console:

1. Go to **IAM & Admin > Service Accounts**
2. Click **Create Service Account**
3. Name it, for example: `terraform-lab-service`
4. Assign the **Editor** or **Compute Admin** role
5. Click **Done**
6. Under the new service account, click **Add Key > Create New Key > JSON**
7. Download the key file to a secure location (for example, `/Users/<username>/Downloads/terraform-key.json`)

Set the environment variable in the terminal:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/<username>/Downloads/terraform-key.json"
```

Verify authentication:

```bash
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
gcloud auth list
```

---

## Step 3: Create a Terraform Working Directory

Create and move into a folder for the lab:

```bash
mkdir ~/terraform-gcp-lab
cd ~/terraform-gcp-lab
```

Or use an existing folder in your project directory.

---

## Step 4: Write the Terraform Configuration

Create a file named `main.tf` and add the following code:

```hcl
provider "google" {
  project = "your-project-id"
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
```

---

## Step 5: Initialize Terraform

From your Terraform working directory, run:

```bash
terraform init
```

This downloads the necessary provider plugins and prepares the environment.

---

## Step 6: Plan and Apply the Configuration

Preview what Terraform will do:

```bash
terraform plan
```

Apply the configuration:

```bash
terraform apply
```

Type **yes** when prompted. Terraform will create the VM in your GCP project.

Verify the VM in the GCP Console under **Compute Engine > VM instances**.

---

## Step 7: Modify the Configuration

To modify resources, update `main.tf`. For example, change the machine type and add labels:

```hcl
resource "google_compute_instance" "demo_vm" {
  name         = "terraform-demo-vm"
  machine_type = "e2-micro"
  zone         = "us-central1-a"

  labels = {
    environment = "dev"
    owner       = "terraform-lab"
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 12
    }
  }

  network_interface {
    network = "default"
  }
}
```

Apply the update:

```bash
terraform apply
```

Terraform will detect and apply only the differences.

---

## Step 8: Add Additional Resources (Optional)

Add a Cloud Storage bucket to the configuration:

```hcl
resource "google_storage_bucket" "lab_bucket" {
  name          = "terraform-lab-bucket-unique-id"
  location      = "us-central1"
  force_destroy = true
}
```

Re-apply the configuration:

```bash
terraform apply
```

Verify the bucket in **Cloud Storage > Buckets** in the GCP Console.

---

## Step 9: Destroy Resources

To remove all resources created by Terraform:

```bash
terraform destroy
```

Confirm with **yes** when prompted. Terraform will safely delete all managed resources.

After destruction, billing for those resources stops immediately.

---

## Step 10: Understanding Terraform State

Terraform maintains a `terraform.tfstate` file that stores the current state of managed resources.

- The file is created after running `terraform apply`
- It must not be edited manually
- In production, store it securely in a remote backend (for example, a GCS bucket or Terraform Cloud)

Key files and folders include:

| File / Folder | Purpose |
|----------------|----------|
| `main.tf` | The configuration file defining resources |
| `terraform.tfstate` | JSON file storing current resource state |
| `.terraform/` | Directory containing provider plugins and metadata |

---

## Step 11: Clean Up

Ensure all resources are destroyed and billing has stopped:

```bash
terraform destroy
```

Then verify in the GCP Console that there are no active VMs, disks, or storage buckets remaining.

---

## References

- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [GCP Provider Guide](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Google Cloud Free Tier](https://cloud.google.com/free)
