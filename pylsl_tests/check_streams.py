"""List available EEG streams, save their info and data to files, and only print stream names."""

from pylsl import StreamInlet, resolve_byprop


def main():
    print("Looking for EEG streams...")
    streams = resolve_byprop("type", "EEG")

    if not streams:
        print("No EEG streams found.")
        return

    print(f"Found {len(streams)} EEG stream(s):\n")

    # Save stream info
    with open("lsl_stream_log.xml", "w") as logfile:
        for i, stream in enumerate(streams):
            name = stream.name()
            print(f"Stream {i+1} name: {name}")  # ✅ ONLY this goes to terminal
            logfile.write(f"--- Stream {i+1} ---\n")
            logfile.write(stream.as_xml() + "\n\n")

    # Create inlet and start logging data to file
    inlet = StreamInlet(streams[0])
    with open("EEG_data.txt", "w") as datafile:
        print(f"\nNow pulling data from: {streams[0].name()}...\n")
        while True:
            sample, timestamp = inlet.pull_sample()
            datafile.write(f"{timestamp}, {sample}\n")


if __name__ == "__main__":
    main()
