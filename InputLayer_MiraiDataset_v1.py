"""
Name: Liam Keefe
Subject: For ECE6612 Project
Date Created: 3/30/2025
Date Last Modified: 5/6/2025
Description: INPUT layer script for the basic Mirai dataset.
This script processes the Mirai dataset and outputs the RMSE values to a CSV
file using the Kitsune ML-NIDS model.
"""
from scipy.stats import norm
from matplotlib import pyplot as plt
import numpy as np
import time
import csv
import os
import sys
import logging

"""
# Kitsune a lightweight online network intrusion detection system based on an
# ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An
# Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and
# detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features),
# and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of
# incremental damped statistics (see the NDSS paper for more details)

# The runtimes presented in the paper, are based on the C++ implimentation
# (roughly 100x faster than the python implimentation)
"""

# SETUP LOGGING FRAMEWORK


def setup_logging(log_directory: str, log_file: str) -> logging.Logger:
    """_summary_
    Configures the logging framework for the script. Returns logger object.

    Args:
        log_directory (str): output directory for the log file.
        log_file (str): name of the log file to append/create.

    Returns:
        logging.Logger: logger object for logging messages.
    """

    log_file_path = os.path.join(log_directory, log_file)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='a')

    return logging.getLogger(__name__)

# IMPORT KITSUNE MODULE


def import_Kitsune_module(path: str, logger: logging.Logger) -> None:
    """_summary_
    Imports the Kitsune module from the specified path.

    Args:
        path (str): path to the Kitsune.py module.
        logger (logging.Logger): object for logging
    """

    script_dir = os.path.dirname(os.path.abspath('Kitsune.py'))
    parent_dir = os.path.join(script_dir, os.pardir)

    kitsune_mod = os.path.abspath(parent_dir)

    if kitsune_mod not in sys.path:
        sys.path.append(kitsune_mod)
        logger.debug(f"Added '{kitsune_mod}' to sys.path")


# Parse packets and compute RMSEs


def execute_packet_processing(K: object, packet_limit: int,
                              logger: logging.Logger) -> list[float]:
    """_summary_
    Processes packets using the Kitsune model and logs the results.

    Args:
        K (object): Kitsune object for processing packets.
        packet_limit (int): maximum number of packets to process.
        logger (logging.Logger): object for logging

    Retuirns:
        list[float]: list of RMSE values for each processed packet.
    """

    # Structure of the output variable
    logger.info("Running Kitsune:")
    RMSEs = []
    i = 0
    start = time.time()

    # Iterative approach to process each packet
    try:
        while True:
            i += 1
            if i % 1000000 == 0:
                logger.info("Processing packet: " + str(i) + " / " +
                            str(packet_limit))
            rmse = K.proc_next_packet()
            if rmse == -1:
                break
            RMSEs.append(rmse)
    except Exception as e:
        logger.critical(f"Error processing packet {i}: {e}")

    stop = time.time()
    logger.info("Packet Processing Complete. Time elapsed: " +
                str(stop - start))

    return RMSEs

# OUTPUTING THE RESULTS


def output_results(RMSEs: list[float], file_name: str, file_dir: str,
                   logger: logging.Logger) -> None:
    """_summary_
    Outputs the RMSE values to a CSV file.

    Args:
        RMSEs (list[float]): list of RMSE values to output.
        file_name (str): name of the output file.
        file_dir (str): directory for the output file.
        logger (logging.Logger): object for logging
    """

    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)

    full_path = os.path.join(file_dir, file_name)\
        if file_dir else file_name

    try:
        with open(full_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["RMSE"])
            for rmse in RMSEs:
                writer.writerow([rmse])
        logger.info(f"RMSE values saved to {full_path}")
    except IOError:
        logger.error("I/O error while writing to file '{full_path}' : {e}")


# Here we demonstrate how one can fit the RMSE scores to a log-normal
# distribution (useful for finding/setting a cutoff threshold \phi)


def output_graph_RMSEs(RMSEs: list[float], FMgrace: int, ADgrace: int,
                       logger: logging.Logger) -> None:
    """_summary_
    Outputs the RMSE values/density to a graph.

    Args:
        RMSEs (list[float]): list of RMSE values to output.
        FMgrace (int): number of instances taken to learn the feature mapping.
        ADgrace (int): number of instances used to train the anomaly detector.
        logger (logging.Logger): object for logging
    """

    benignSample = np.log(RMSEs[FMgrace+ADgrace+1:100000])
    logProbs = norm.logsf(np.log(RMSEs),
                          np.mean(benignSample),
                          np.std(benignSample))

    # plot the RMSE anomaly scores
    logger.info("Plotting RMSE anomaly scores")
    plt.figure(figsize=(10, 5))
    plt.scatter(range(FMgrace+ADgrace+1, len(RMSEs)),
                RMSEs[FMgrace+ADgrace+1:],
                s=0.1,
                c=logProbs[FMgrace+ADgrace+1:],
                cmap='RdYlGn')
    plt.yscale("log")
    plt.title("Anomaly Scores from Kitsune's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Packet Number")
    figbar = plt.colorbar()
    figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
    plt.savefig("Kitsune_MiraiDataset.png", dpi=300)


def main():
    log_directory = 'log'
    log_file = "InputLayer_LogFile.txt"

    logger = setup_logging(log_directory, log_file)
    logger.info("Setup Logging Input Layer Script - Mirai Dataset")

    path_Kitsune = '../Kitsune.py'
    import_Kitsune_module(path_Kitsune, logger)
    logger.info("Attempting to set up path for Kitsune module...")
    try:
        from Kitsune import Kitsune
        logger.info("Successfully imported Kitsune module.")

    except ImportError:
        logger.error("Failed to import Kitsune after path modification.")

    # Input file and # of packets
    path = "mirai.pcap"  # the pcap, pcapng, or tsv file to process.
    packet_limit = np.inf  # the number of packets to process

    # KitNET params:
    maxAE = 10  # maximum size for any autoencoder in the ensemble layer
    FMgrace = 5000  # the # of inst. taken to learn (ensemble's architecture)
    ADgrace = 50000  # the # of inst. used to train (ensemble itself)

    # Build Kitsune
    K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

    # Process incoming packets and parses; computes RMSEs
    RMSE_out = execute_packet_processing(K, packet_limit, logger)

    filename = "output_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    output_dir = "output_mirai"

    # Output the RMSE values to a CSV file
    output_results(RMSE_out, filename, output_dir, logger)
    logger.info("RMSE values outputted successfully")

    # Output the RMSE values/density to a graph;
    output_graph_RMSEs(RMSE_out, FMgrace, ADgrace, logger)
    logger.info("RMSE graph outputted successfully")


if __name__ == "__main__":
    main()
