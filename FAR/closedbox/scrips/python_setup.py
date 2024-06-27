import os
import subprocess
import time

I = ['H1L1', 'H1V1', 'L1V1', 'H1L1V1']
for ifos in I:
    # Define the content of the Bash script
    bash_script_content = f"""#!/bin/bash
cd /data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/FAR/
# Run the Python script with the specified argument
python3 /data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/FAR/roc_main.py --ifos={ifos}
"""

    # Define the name of the Bash script to be created
    bash_script_filename = f"run_roc_curves_{ifos}.sh"

    # Write the content to the Bash script file
    try:
        with open(bash_script_filename, 'w') as file:
            file.write(bash_script_content)
        # Make the Bash script executable
        #os.chmod(bash_script_filename, 0o755)
    except IOError as e:
        print(f"Error writing to {bash_script_filename}: {e}")
        continue

    # Define the qsub command to submit the job
    qsub_command = [
        "qsub",
        "-q", "short",
        "-o", f"output_{ifos}.log",
        "-j", "oe",
        bash_script_filename
    ]

    # Submit the job using qsub
    try:
        subprocess.run(qsub_command, check=True)
        print(f"Bash script '{bash_script_filename}' has been created and submitted to qsub.")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job with qsub: {e}")

    time.sleep(5)
