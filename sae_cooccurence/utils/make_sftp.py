import re

def convert_ssh_to_sftp(ssh_command):
    # Regular expression pattern to match the SSH command format
    pattern = r'^ssh\s+(\S+@\S+)\s+-p\s+(\d+)\s+-i\s+(\S+)$'
    
    # Use regular expression to extract the relevant parts from the SSH command
    match = re.match(pattern, ssh_command)
    
    if match:
        # Extract the captured groups from the match
        user_host = match.group(1)
        port = match.group(2)
        identity_file = match.group(3)
        
        # Construct the SFTP command
        sftp_command = f'sftp -i {identity_file} -P {port} {user_host}'
        
        return sftp_command
    else:
        return None

# Example usage
ssh_command = 'ssh root@71.158.89.73 -p 4410 -i ~/.ssh/id_ed25519'
sftp_command = convert_ssh_to_sftp(ssh_command)

if sftp_command:
    print("SFTP command:")
    print(sftp_command)
else:
    print("Invalid SSH command format.")