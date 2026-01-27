"""
Test script to verify the setup is working correctly.
Tests GPU/CPU availability, TensorFlow installation, and basic functionality.
"""
import sys
import os

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.common import setupGpu, ensureDirs
else:
    from .common import setupGpu, ensureDirs

def testTensorflow():
    """Test TensorFlow installation and basic operations."""
    print("\n" + "="*80)
    print("TESTING TENSORFLOW")
    print("="*80)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")

        # Test basic operation
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
        print(f"✓ Basic TensorFlow operations work: [1,2,3] + [4,5,6] = {c.numpy()}")

        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False


def testGpuSetup():
    """Test GPU setup and availability."""
    print("\n" + "="*80)
    print("TESTING GPU SETUP")
    print("="*80)

    try:
        hasGpu = setupGpu()
        if hasGpu:
            print("✓ GPU is available and configured")
        else:
            print("✓ No GPU available, using CPU (this is okay)")
        return True
    except Exception as e:
        print(f"✗ GPU setup test failed: {e}")
        return False


def testDataLoading():
    """Test data loading functionality."""
    print("\n" + "="*80)
    print("TESTING DATA LOADING")
    print("="*80)

    try:
        from basil_core.data.mnist import loadMnist, makeLoaders

        print("Loading MNIST dataset...")
        train, test = loadMnist()
        print(f"✓ MNIST loaded: {len(train)} train samples, {len(test)} test samples")

        print("Creating data loaders...")
        trainLoaders, testLoader = makeLoaders(train, test, batchSize=32, nClients=3)
        print(f"✓ Data loaders created: {len(trainLoaders)} clients, {len(testLoader)} test batches")

        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def testModelCreation():
    """Test model creation."""
    print("\n" + "="*80)
    print("TESTING MODEL CREATION")
    print("="*80)

    try:
        from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
        import tensorflow as tf

        # Test MNIST model
        mnistModel = MNISTModel()
        testInput = tf.random.normal((2, 28, 28))
        output = mnistModel(testInput, training=False)
        print(f"✓ MNIST model created and works: input shape (2, 28, 28) -> output shape {output.shape}")

        # Test CIFAR model
        cifarModel = CIFARModel()
        testInput = tf.random.normal((2, 32, 32, 3))
        output = cifarModel(testInput, training=False)
        print(f"✓ CIFAR-10 model created and works: input shape (2, 32, 32, 3) -> output shape {output.shape}")

        # Test N-MNIST model
        nmnistModel = NMNISTModel()
        testInput = tf.random.normal((2, 34, 34))
        output = nmnistModel(testInput, training=False)
        print(f"✓ N-MNIST model created and works: input shape (2, 34, 34) -> output shape {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def testTraining():
    """Test basic training functionality."""
    print("\n" + "="*80)
    print("TESTING BASIC TRAINING")
    print("="*80)

    try:
        from basil_core.data.mnist import loadMnist, makeLoaders
        from basil_core.models import MNISTModel
        from basil_core.basil import BasilNode, basilRingTrainingWithAttack

        # Load small dataset
        print("Loading data...")
        train, test = loadMnist()
        trainLoaders, testLoader = makeLoaders(train, test, batchSize=32, nClients=2)

        # Create 2 nodes
        print("Creating nodes...")
        nodes = []
        for i in range(2):
            model = MNISTModel()
            nodes.append(BasilNode(
                nodeId=i,
                model=model,
                dataLoader=trainLoaders[i],
                S=2,
                noiseModel="none",
                sigma=0.0,
                lr0=0.05,
                localEpochs=1
            ))

        # Train for 2 rounds
        print("Training for 2 rounds (this may take a minute)...")
        avgAccHist, worstAccHist = basilRingTrainingWithAttack(
            nodes=nodes,
            rounds=2,
            testLoader=testLoader,
            attackTypes=["none"],
            attackerIds=[],
            sigma=0.0,
            noiseModel="none",
            lr0=0.05,
            stepsPerEpoch=10  # Limit steps for quick test
        )

        print(f"✓ Training completed successfully")
        print(f"  Round 0: avg_acc={avgAccHist[0]:.4f}")
        print(f"  Round 1: avg_acc={avgAccHist[1]:.4f}")
        print(f"  Round 2: avg_acc={avgAccHist[2]:.4f}")

        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def testWcmAvailability():
    """Test WCM module availability."""
    print("\n" + "="*80)
    print("TESTING WCM MODULE")
    print("="*80)

    try:
        from noise_comm.wcm import wcmStep, sampleBoundaryPayload, scaSurrogateLoss
        print("✓ WCM module is available")
        print("  Available functions: wcmStep, sampleBoundaryPayload, scaSurrogateLoss")
        return True
    except ImportError as e:
        print(f"⚠ WCM module not available: {e}")
        print("  This is okay - WCM will fall back to standard training")
        return True  # Not a failure, just a warning


def runAllTests():
    """Run all tests and report results."""
    print("\n" + "#"*80)
    print("# SETUP VERIFICATION TEST SUITE")
    print("#"*80)

    ensureDirs()  # Ensure directories exist

    tests = [
        ("TensorFlow Installation", testTensorflow),
        ("GPU Setup", testGpuSetup),
        ("Data Loading", testDataLoading),
        ("Model Creation", testModelCreation),
        ("WCM Module", testWcmAvailability),
        ("Basic Training", testTraining),
    ]

    results = []
    for testName, testFunc in tests:
        try:
            success = testFunc()
            results.append((testName, success))
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR in {testName}: {e}")
            results.append((testName, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for testName, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {testName}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED! Setup is working correctly.")
        print("You can now run experiments with: python scripts/runComprehensiveTest.py --quick")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = runAllTests()
    sys.exit(0 if success else 1)
