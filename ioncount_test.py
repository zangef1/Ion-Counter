import MDAnalysis as mda
import numpy as np
from pathlib import Path
from datetime import datetime

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
        
        # Tracking
        self.ion_states = {}  # Track ion positions
        self.last_permeation_frame = {}
        self.permeation_events = []
        
        # Parameters for detection
        self.gate_margin = 3.0
        self.debug_mode = True
        
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
        
        return upper_center, lower_center

    def _is_within_cylinder(self, point, gate1_center, gate2_center):
        """Check if point is within the cylindrical channel region."""
        # Vector from point to channel axis
        v = point - self.channel_center
        
        # Project onto channel axis
        proj = np.dot(v, self.channel_axis) * self.channel_axis
        
        # Get radial component
        radial = v - proj
        radial_dist = np.linalg.norm(radial)
        
        # Check cylinder radius
        if radial_dist > self.channel_radius:
            return False
            
        # Check if between gates
        axial_pos = np.dot(v, self.channel_axis)
        return abs(axial_pos) <= self.channel_length/2

    def _check_ion_position(self, ion_id, ion_pos, upper_center, lower_center, frame):
        """Track ion position relative to gates and check for permeation."""
        if ion_id not in self.ion_states:
            self.ion_states[ion_id] = {
                'last_gate': None,
                'sequence': [],
                'last_z': None,
                'permeation_started': False,  # Track if permeation has started
                'permeation_frame': None,     # Frame where permeation was detected
                'waiting_for_exit': False,    # Track if waiting for ion to exit
                'last_in_cylinder': False     # Track if ion was in cylinder last frame
            }
        
        # Get ion's position along channel axis
        ion_vec = ion_pos - self.channel_center
        ion_z = np.dot(ion_vec, self.channel_axis)
        
        # Get gate positions along axis
        upper_z = np.dot(upper_center - self.channel_center, self.channel_axis)
        lower_z = np.dot(lower_center - self.channel_center, self.channel_axis)
        
        # Store z position for tracking
        if self.ion_states[ion_id]['last_z'] is None:
            self.ion_states[ion_id]['last_z'] = ion_z
            return
            
        # Check if ion is in cylinder
        in_cylinder = self._is_within_cylinder(ion_pos, upper_center, lower_center)
        
        # If we're waiting for exit and ion is no longer in cylinder
        if self.ion_states[ion_id]['waiting_for_exit']:
            if not in_cylinder and self.ion_states[ion_id]['last_in_cylinder']:
                # Ion has exited - record the complete permeation
                permeation_frame = self.ion_states[ion_id]['permeation_frame']
                self.permeation_events.append({
                    'ion_id': ion_id,
                    'start_frame': permeation_frame,
                    'exit_frame': frame
                })
                if self.debug_mode:
                    print(f"\nComplete permeation detected:")
                    print(f"Ion {ion_id} permeated at frame {permeation_frame}")
                    print(f"Exited channel at frame {frame}")
                    print(f"Total permeation time: {frame - permeation_frame} frames")
                
                # Reset tracking
                self.ion_states[ion_id]['waiting_for_exit'] = False
                self.ion_states[ion_id]['permeation_frame'] = None
                self.last_permeation_frame[ion_id] = frame
                
            self.ion_states[ion_id]['last_in_cylinder'] = in_cylinder
            return
            
        # Normal permeation detection
        current_gate = None
        
        # Check upper gate crossing
        if (self.ion_states[ion_id]['last_z'] < upper_z and ion_z > upper_z) or \
           abs(ion_z - upper_z) < self.gate_margin:
            current_gate = 'upper'
            
        # Check lower gate crossing
        elif (self.ion_states[ion_id]['last_z'] < lower_z and ion_z > lower_z) or \
             abs(ion_z - lower_z) < self.gate_margin:
            current_gate = 'lower'
        
        # Update sequence if gate crossed
        if current_gate and current_gate != self.ion_states[ion_id]['last_gate']:
            # Check if ion is within cylinder radius
            if in_cylinder:
                self.ion_states[ion_id]['sequence'].append(current_gate)
                self.ion_states[ion_id]['last_gate'] = current_gate
                
                # Debug output
                if self.debug_mode and frame % 100 == 0:
                    print(f"\nFrame {frame} - Ion {ion_id}:")
                    print(f"Position: {ion_z:.2f}")
                    print(f"Gates: Upper={upper_z:.2f}, Lower={lower_z:.2f}")
                    print(f"Sequence: {self.ion_states[ion_id]['sequence']}")
                
                # Check for permeation
                sequence = self.ion_states[ion_id]['sequence']
                if len(sequence) >= 2:
                    # Forward permeation: upper -> lower
                    if sequence[-2:] == ['upper', 'lower']:
                        # Check frame gap
                        if ion_id not in self.last_permeation_frame or \
                           frame - self.last_permeation_frame[ion_id] >= self.min_frame_gap:
                            # Start tracking exit
                            self.ion_states[ion_id]['waiting_for_exit'] = True
                            self.ion_states[ion_id]['permeation_frame'] = frame
                            self.ion_states[ion_id]['last_in_cylinder'] = True
                            if self.debug_mode:
                                print(f"\nPotential permeation detected - Ion {ion_id} at frame {frame}")
                                print(f"Waiting for ion to exit channel...")
                        self.ion_states[ion_id]['sequence'] = []
        
        # Update last z position
        self.ion_states[ion_id]['last_z'] = ion_z
        
        # Update cylinder status if waiting for exit
        if self.ion_states[ion_id]['waiting_for_exit']:
            self.ion_states[ion_id]['last_in_cylinder'] = in_cylinder

    def run_analysis(self):
        """Run analysis on trajectory."""
        print(f"Starting analysis...")
        print(f"Channel radius: {self.channel_radius} Å")
        print(f"Gate margin: {self.gate_margin} Å")
        print(f"Min frame gap: {self.min_frame_gap}")
        
        for ts in self.u.trajectory:
            if ts.frame % 1000 == 0:
                print(f"Processing frame {ts.frame}")
                
            # Get gate positions
            upper_center, lower_center = self._get_gate_centers(ts)
            
            # Check all ions
            ions = self.u.select_atoms(self.ion_selection)
            for ion in ions:
                self._check_ion_position(ion.resid, ion.position, upper_center, lower_center, ts.frame)

    def print_results(self):
        """Print permeation events with complete timing information."""
        print("\nForward Permeation Events:")
        print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
        print("-" * 55)
        for event in self.permeation_events:
            total_time = event['exit_frame'] - event['start_frame']
            print(f"{event['ion_id']:6d} | {event['start_frame']:11d} | {event['exit_frame']:10d} | {total_time:10d}")
        
        print(f"\nTotal forward permeation events: {len(self.permeation_events)}")

def main():
    """Main function to run the analysis."""
    top_file = "/home/data/Yongcheng/GIRK4/CHL/com_4fs.prmtop"
    traj_file = "/home/data/Yongcheng/GIRK4/CHL/G4_CHL_PIP2_1us.trr"
    
    # Define gate residues
    upper_gate_residues = [138, 788, 463, 1113]
    lower_gate_residues = (
        list(range(259, 266)) +
        list(range(909, 916)) +
        list(range(584, 591)) +
        list(range(1234, 1241))
    )
    
    try:
        print(f"Loading system...")
        u = mda.Universe(top_file, traj_file)
        
        analyzer = IonPermeationAnalysis(
            universe=u,
            ion_selection="resname K+",
            upper_gate_residues=upper_gate_residues,
            lower_gate_residues=lower_gate_residues,
            channel_radius=8.0,
            min_frame_gap=5
        )
        
        analyzer.run_analysis()
        analyzer.print_results()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()