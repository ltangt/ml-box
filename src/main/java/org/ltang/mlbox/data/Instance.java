package org.ltang.mlbox.data;

/**
 * Training data Instance
 * @author Liang Tang (ltang@linkedin.com)
 */
public class Instance {

  private final SparseVector _features;

  private final float _label;

  private final float _weight;

  private final float _offset;

  private final int _flag;


  public Instance(final SparseVector f, final float label) {
    this(f, label, 1f);
  }

  public Instance(final SparseVector f, final float label, final float weight) {
    this(f, label, weight, 0f, 0);
  }

  public Instance(final SparseVector f, final float label, final float weight, final float offset, final int flag) {
    _features = f;
    _label = label;
    _flag = flag;
    _weight = weight;
    _offset = offset;
  }

  public SparseVector getFeatures() {
    return _features;
  }

  public float getLabel() {
    return _label;
  }

  public float getWeight() {
    return _weight;
  }

  public float getOffset() {
    return _offset;
  }

  public int getFlag() {
    return _flag;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    Instance instance = (Instance) o;

    if (Float.compare(instance._label, _label) != 0) {
      return false;
    }
    if (Float.compare(instance._weight, _weight) != 0) {
      return false;
    }
    if (Float.compare(instance._offset, _offset) != 0) {
      return false;
    }
    if (_flag != instance._flag) {
      return false;
    }
    return !(_features != null ? !_features.equals(instance._features) : instance._features != null);
  }

  @Override
  public int hashCode() {
    int result = _features != null ? _features.hashCode() : 0;
    result = 31 * result + (_label != +0.0f ? Float.floatToIntBits(_label) : 0);
    result = 31 * result + (_weight != +0.0f ? Float.floatToIntBits(_weight) : 0);
    result = 31 * result + (_offset != +0.0f ? Float.floatToIntBits(_offset) : 0);
    result = 31 * result + _flag;
    return result;
  }

  @Override
  public String toString() {
    return "Instance{" +
        "_features=" + _features +
        ", _label=" + _label +
        ", _weight=" + _weight +
        ", _offset=" + _offset +
        ", _flag=" + _flag +
        '}';
  }
}
