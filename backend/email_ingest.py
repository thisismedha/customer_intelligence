import os
import base64
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from datetime import datetime
import re

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Brands to filter (modify this list)
TARGET_BRANDS = [
    'target',
    'bestbuy',
    'walmart',
    'bananarepublicfactory',
    'VictoriasSecret',
    'oldnavy'

]

def authenticate_gmail():
    """Authenticate and return Gmail API service"""
    creds = None
    
    # Token file stores user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def get_promotional_emails(service, brands, max_results=1000):
    """Fetch promotional emails from specific brands"""
    emails_data = []
    
    # Build query for promotional category and specific brands
    brand_query = ' OR '.join([f'from:{brand}' for brand in brands])
    query = f'category:promotions ({brand_query})'
    
    try:
        # Get list of messages
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            print('No promotional emails found from specified brands.')
            return emails_data
        
        print(f'Found {len(messages)} emails. Processing...')
        
        for msg in messages:
            # Get full message details
            message = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            
            email_data = parse_email(message)
            emails_data.append(email_data)
        
        return emails_data
    
    except Exception as e:
        print(f'An error occurred: {e}')
        return emails_data

def parse_email(message):
    """Parse email message and extract relevant information"""
    headers = message['payload']['headers']
    
    # Extract header information
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
    date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
    
    # Extract email body
    body = get_email_body(message['payload'])
    
    email_data = {
        'id': message['id'],
        'thread_id': message['threadId'],
        'subject': subject,
        'from': sender,
        'date': date,
        'snippet': message.get('snippet', ''),
        'body': body,
        'labels': message.get('labelIds', [])
    }
    
    return email_data

def get_email_body(payload):
    """Extract email body from payload"""
    body = ''
    
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    body = base64.urlsafe_b64decode(
                        part['body']['data']
                    ).decode('utf-8')
                    break
            elif part['mimeType'] == 'text/html' and not body:
                if 'data' in part['body']:
                    body = base64.urlsafe_b64decode(
                        part['body']['data']
                    ).decode('utf-8')
    else:
        if 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(
                payload['body']['data']
            ).decode('utf-8')
    
    return body

def save_to_json(emails, filename='promotional_emails.json'):
    """Save emails to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)
    print(f'Saved {len(emails)} emails to {filename}')

def save_to_csv(emails, filename='promotional_emails.csv'):
    """Save emails to CSV file"""
    import csv
    
    if not emails:
        print('No emails to save.')
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=emails[0].keys())
        writer.writeheader()
        writer.writerows(emails)
    
    print(f'Saved {len(emails)} emails to {filename}')

def main():
    """Main function"""
    print('Gmail Promotional Email Ingestion Script')
    print('=' * 50)
    
    # Authenticate
    print('\nAuthenticating with Gmail API...')
    service = authenticate_gmail()
    print('Authentication successful!')
    
    # Fetch emails
    print(f'\nFetching promotional emails from: {", ".join(TARGET_BRANDS)}')
    emails = get_promotional_emails(service, TARGET_BRANDS, max_results=1000)
    
    if emails:
        # Display summary
        print('\n' + '=' * 50)
        print(f'Total emails fetched: {len(emails)}')
        print('\nSample email:')
        print(f'Subject: {emails[0]["subject"]}')
        print(f'From: {emails[0]["from"]}')
        print(f'Date: {emails[0]["date"]}')
        
        # Save to files
        save_to_json(emails)
        save_to_csv(emails)
    else:
        print('No emails found.')

if __name__ == '__main__':
    main()