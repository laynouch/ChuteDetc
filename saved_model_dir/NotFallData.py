import firebase_admin
from firebase_admin import credentials, firestore
import csv
# Path to your service account key file
service_account_key_path = 'C:/Users/Lina/Desktop/ES DATA.json'

# Use a service account
cred = credentials.Certificate(service_account_key_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

# Optional: Print a message to confirm initialization
print("Firebase Admin SDK initialized successfully")

# Reference to a collection
users_ref = db.collection('Not_Fall')

# Fetch all documents in the collection
docs = users_ref.stream()


print('--------------------------------------------------------------------------')

fields = [
    'acc_max','gyro_max','acc_kurtosis','gyro_kurtosis','lin_max','acc_skewness',
    'gyro_skewness','post_gyro_max','post_lin_max','fall'

]

# Define the output CSV file name
csv_file = 'NotFallData.csv'

# Open CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    
    # Iterate over the documents and write to CSV
    for doc in docs:
        data = doc.to_dict()
        extracted_data = {field: data.get(field, None) for field in fields}
        writer.writerow(extracted_data)

print(f"Data successfully written to {csv_file}")