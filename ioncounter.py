import MDAnalysis as mda
import numpy as np
from collections import Counter
from tqdm import tqdm  # Import tqdm for the progress bar
import argparse

permeation_events = []  # List of dictionaries to track events

class IonPermeationAnalysis:
    def __init__(self, 
                 universe: mda.Universe,
                 ion_selection: str,
                 upper_gate_residues: list,
                 lower_gate_residues: list,
                 channel_radius: float = 8.0,
                 min_frame_gap: int = 5):
        
        self.u = universe
        self.ion_selection = ion_selection
        self.min_frame_gap = min_frame_gap
        self.channel_radius = channel_radius
        
        # Gate definitions
        self.upper_gate_residues = upper_gate_residues
        self.lower_gate_residues = lower_gate_residues
        
        # Dictionary to track ion states
        self.ion_states = {}
        
    def _get_gate_centers(self, ts):
        """Get centers of mass for both gates."""
        upper_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.upper_gate_residues))}")
        lower_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.lower_gate_residues))}")
        
        upper_center = upper_sel.center_of_mass()
        lower_center = lower_sel.center_of_mass()
        
        # Calculate channel axis and center
        self.channel_vector = lower_center - upper_center
        self.channel_length = np.linalg.norm(self.channel_vector)
        self.channel_axis = self.channel_vector / self.channel_length
        self.channel_center = (upper_center + lower_center) / 2

        #print(f"Frame {ts.frame}: Computed gate centers")
        return upper_center, lower_center

    def _is_within_cylinder(self, v):
        """Check if point is within the cylindrical channel region."""
        proj = np.dot(v, self.channel_axis) * self.channel_axis
        radial = v - proj
        radial_dist = np.linalg.norm(radial)

        # Check cylinder radius
        if radial_dist > self.channel_radius:
            return False

        # Check if between gates
        axial_pos = np.dot(v, self.channel_axis)
        return abs(axial_pos) <= self.channel_length / 2

    def _check_ion_position(self, ion_id, ion_pos, upper_center, lower_center, frame):
        """Track ion position relative to gates and check for permeation."""

        # Compute ion position relative to the channel center
        ion_vec = ion_pos - self.channel_center
        ion_z = np.dot(ion_vec, self.channel_axis)

        if ion_id not in self.ion_states:
            self.ion_states[ion_id] = {'upper_flag': 0, 'lower_flag': 0, 'upper_flag_frame': 0, 'lower_flag_frame': 0, 'prev_ion_z': None}

        # Get gate positions along the channel axis
        upper_z = np.dot(upper_center - self.channel_center, self.channel_axis)
        lower_z = np.dot(lower_center - self.channel_center, self.channel_axis)

        # Check if ion is inside the cylinder before updating flags
        in_cylinder = self._is_within_cylinder(ion_vec)
        
    
        if in_cylinder: 
            if self.ion_states[ion_id]['upper_flag'] == 0:
                #if ion_id == 2554:
                #    print(frame, self.ion_states[ion_id])
                #first time in the cylinder
                if self.ion_states[ion_id]['prev_ion_z']:   # not including ions already in the cylinder in the beggining of simulation
                    if self.ion_states[ion_id]['prev_ion_z'] < upper_z:
                        self.ion_states[ion_id]['upper_flag'] = 1
                        #print(f"Frame {frame}: Ion {ion_id} crossed upper gate, setting upper_flag = 1")
                        if self.ion_states[ion_id]['upper_flag_frame'] == 0 :
                            self.ion_states[ion_id]['upper_flag_frame'] = frame
            elif self.ion_states[ion_id]['upper_flag'] == 1 and self.ion_states[ion_id]['lower_flag'] == 1:
                #2 scenarios
                if self.ion_states[ion_id]['prev_ion_z'] < upper_z:
                    #ion permeates again
                    start_frame = self.ion_states[ion_id]['upper_flag_frame']
                    exit_frame = self.ion_states[ion_id]['lower_flag_frame']
                    total_time = exit_frame - start_frame
                    permeation_events.append({
                            'ion_id': ion_id,
                            'start_frame': start_frame,
                            'exit_frame': exit_frame,
                            'total_time': total_time
                    })
                    self.ion_states[ion_id]['upper_flag_frame'] = frame
                # ion reenters from lower gate
                self.ion_states[ion_id]['lower_flag'] = 0

                    #self.ion_states[ion_id]['upper_flag'] = 1
                    #already in cylinder, make sure lower flag is 0
                    #elif prev_ion_z > 0:
                    #   self.ion_states[ion_id]['lower_flag'] = 0
                    #if ion_id == 2554:
                    #    print(frame, self.ion_states[ion_id])
            

        # If the ion crosses the lower gate downward and upper_flag is 1, set lower_flag = 1
        if ion_z > lower_z and self.ion_states[ion_id]['upper_flag'] == 1:
            #print(f"Frame {frame}: Ion {ion_id} crossed lower gate, setting lower_flag = 1")
            if self.ion_states[ion_id]['lower_flag'] == 0 :
                self.ion_states[ion_id]['lower_flag_frame'] = frame
                self.ion_states[ion_id]['lower_flag'] = 1
            #if ion_id == 2554:
            #        print(frame, self.ion_states[ion_id])

        # if ion goes in cylinder throught upper gate and then leaves, if it leaves the cylinder make upper_flag 0
        if not in_cylinder and self.ion_states[ion_id]['upper_flag'] == 1 and self.ion_states[ion_id]['lower_flag'] == 0:
            self.ion_states[ion_id]['upper_flag'] = 0

        self.ion_states[ion_id]['prev_ion_z'] = ion_z
        
    def run_analysis(self):
        """Run analysis on trajectory."""
        print("Starting analysis...")


        # Get total number of frames
        total_frames = len(self.u.trajectory)

        for ts in tqdm(self.u.trajectory, total=total_frames, desc="Processing Frames", unit="frame"):
            #if ts.frame % 1000 == 0:
            #    print(f"Processing frame {ts.frame}")
                
            # Get gate positions
            upper_center, lower_center = self._get_gate_centers(ts)
            
            
            # Check all ions
            ions = self.u.select_atoms(self.ion_selection)
            for ion in ions:
                self._check_ion_position(ion.resid, ion.position, upper_center, lower_center, ts.frame)

    def print_results(self):
        """Print permeation events (flags 1,1) with timing details."""
        print("\nFinal Permeation Events (1,1 Flags):")
        print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
        print("-" * 55)

        #permeation_events = []

        for ion_id, state in self.ion_states.items():
            if state['upper_flag'] == 1 and state['lower_flag'] == 1:
                start_frame = state['upper_flag_frame']
                exit_frame = state['lower_flag_frame']
                total_time = exit_frame - start_frame
                permeation_events.append({
                    'ion_id': ion_id,
                    'start_frame': start_frame,
                    'exit_frame': exit_frame,
                    'total_time': total_time
                })

        # Sort events by start frame (optional)
        permeation_events.sort(key=lambda x: x['start_frame'])

        # Print formatted table
        # for event in permeation_events:
        #     print(f"{event['ion_id']:6d} | {event['start_frame']:11d} | {event['exit_frame']:10d} | {event['total_time']:10d}")
        for idx, event in enumerate(permeation_events, start=1):  # `start=1` begins numbering from 1
            print(f"{idx:3d} | {event['ion_id']:6d} | {event['start_frame']:11d} | {event['exit_frame']:10d} | {event['total_time']:10d}")


        print(f"\nTotal forward permeation events: {len(permeation_events)}")
def save_exit_frames_to_txt(filename="exit_frames.txt"):
    """Save all exit_frame values in a simple array format to a text file."""
    with open(filename, 'w') as f:
        f.write("[")  # Begin array
        f.write(", ".join(str(event['exit_frame']) for event in permeation_events))
        f.write("]")  # End array
    print(f"Exit frames saved to {filename}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Run ion permeation analysis on a molecular dynamics trajectory.")
    parser.add_argument("--top_file", required=True, help="Path to the topology file (e.g., .prmtop)")
    parser.add_argument("--traj_file", required=True, help="Path to the trajectory file (e.g., .trr or .xtc)")
    parser.add_argument("--channel", required=True, choices=["GIRK14_2_2", "GIRK14_1_3", "GIRK4", "GIRK12","Kir_4.1_5.1", "GIRK2"], help="Channel type (GIRK4 or GIRK12)")

    args = parser.parse_args()
    
    channel_residues = {
       	"GIRK2": {
            "upper_gate_residues": [138, 794, 466, 1122],
            "lower_gate_residues": list(range(260, 266)) + list(range(915, 921)) + list(range(587, 593)) + list(range(1243, 1249))
        },
	 "GIRK4": {
            "upper_gate_residues": [138, 788, 463, 1113],
            "lower_gate_residues": list(range(259, 266)) + list(range(909, 916)) + list(range(584, 591)) + list(range(1234, 1241))
        },
        "GIRK12": {
            "upper_gate_residues": [138, 786, 462, 1111],
            "lower_gate_residues": list(range(259, 265)) + list(range(907, 913)) + list(range(583, 589)) + list(range(1232, 1238))
        },
        "Kir_4.1_5.1": {
            "upper_gate_residues": [1112, 136, 462, 787],
            "lower_gate_residues": list(range(582, 586)) + list(range(260, 265)) + list(range(909, 914)) + list(range(1231, 1236))
        },
	"GIRK14_1_3": {
            "upper_gate_residues": [1114, 139, 464, 789],
            "lower_gate_residues": list(range(260, 265)) + list(range(589, 590)) + list(range(910, 915)) + list(range(1235, 1240))
        },
	 "GIRK14_2_2": {
            "upper_gate_residues": [1115, 139, 465, 790],
            "lower_gate_residues": list(range(260, 265)) + list(range(586, 591)) + list(range(911, 916)) + list(range(1236, 1241))
        }

    }
    
    try:
        print("Loading system...")
        u = mda.Universe(args.top_file, args.traj_file)
        
        analyzer = IonPermeationAnalysis(
            universe=u,
            ion_selection="resname K+",
            upper_gate_residues=channel_residues[args.channel]["upper_gate_residues"],
            lower_gate_residues=channel_residues[args.channel]["lower_gate_residues"],
            channel_radius=8.0,
            min_frame_gap=5
        )
        
        analyzer.run_analysis()
        analyzer.print_results()
        save_exit_frames_to_txt()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


