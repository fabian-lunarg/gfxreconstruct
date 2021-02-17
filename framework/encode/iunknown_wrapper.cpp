/*
** Copyright (c) 2021 LunarG, Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
*/

#include "encode/iunknown_wrapper.h"

#include "encode/dx12_object_wrapper_resources.h"
#include "encode/trace_manager.h"
#include "generated/generated_dx12_wrapper_creators.h"

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(encode)

IUnknown_Wrapper::IUnknown_Wrapper(REFIID riid, IUnknown* wrapped_object, DxWrapperResources* resources) :
    riid_(riid), object_(wrapped_object, false), capture_id_(TraceManager::GetUniqueId()), ref_count_(1)
{
    assert(wrapped_object != nullptr);

    // Register this wrapper with the list of related resources, which represents a list of pointers to the same object
    // with different IIDs, all of which share a reference count and will stay active until all of the resources in the
    // list are released by the application.
    if (resources != nullptr)
    {
        resources_ = resources;
        resources_->AddWrapper(this);
        resources_->IncrementSharedCount();
    }
    else
    {
        resources_ = new DxWrapperResources(this);
    }
}

HRESULT IUnknown_Wrapper::QueryInterface(REFIID riid, void** object)
{
    // Check for a query from the capture framework to determine if an object with an IUknown* type is really a
    // wrapped object.  When the IID is a match, return a success code and a pointer to the current object
    // without incrementing the reference count.
    if (IsEqualIID(riid, IID_IUnknown_Wrapper))
    {
        (*object) = this;
        return S_OK;
    }

    HRESULT result     = E_FAIL;
    auto    manager    = TraceManager::Get();
    auto    call_scope = manager->IncrementCallScope();

    if (call_scope == 1)
    {
        result = object_->QueryInterface(riid, object);

        if (SUCCEEDED(result))
        {
            WrapObject(riid, object, resources_);
        }
    }
    else
    {
        result = object_->QueryInterface(riid, object);
    }

    manager->DecrementCallScope();

    return result;
}

ULONG IUnknown_Wrapper::AddRef()
{
    resources_->IncrementSharedCount();
    return ++ref_count_;
}

ULONG IUnknown_Wrapper::Release()
{
    auto shared_count = resources_->DecrementSharedCount();
    auto local_count  = --ref_count_;

    if (shared_count == 0)
    {
        // The resources_ destructor destroys this wrapper and all other wrappers linked to it, so no additional work
        // may be performed in this function, as the current wrapper will no longer be valid.
        delete resources_;
    }

    return local_count;
}

GFXRECON_END_NAMESPACE(encode)
GFXRECON_END_NAMESPACE(gfxrecon)